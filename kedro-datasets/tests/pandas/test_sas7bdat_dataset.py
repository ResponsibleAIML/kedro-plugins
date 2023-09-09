from pathlib import Path, PurePosixPath

import pandas as pd
import pytest
from fsspec.implementations.http import HTTPFileSystem
from fsspec.implementations.local import LocalFileSystem
from gcsfs import GCSFileSystem
from kedro.io import DataSetError
from kedro.io.core import PROTOCOL_DELIMITER, Version
from s3fs.core import S3FileSystem

from kedro_datasets.pandas import Sas7bdatDataset

FILENAME = "test.sas7bdat"


@pytest.fixture
def filepath_sas():
    p = Path(__file__).parent.resolve()
    return (p / FILENAME).as_posix()


@pytest.fixture
def sas_data_set(filepath_sas, load_args, save_args, fs_args):
    return Sas7bdatDataset(
        filepath=filepath_sas,
        load_args=load_args,
        save_args=save_args,
        fs_args=fs_args,
    )


@pytest.fixture
def versioned_sas_data_set(filepath_sas, load_version, save_version):
    return Sas7bdatDataset(
        filepath=filepath_sas, version=Version(load_version, save_version)
    )


@pytest.fixture
def dummy_dataframe(filepath_sas):
    return pd.read_sas(filepath_sas)


class TestSas7bdatDataset:
    def test_credentials_propagated(self, mocker):
        """Test propagating credentials for connecting to GCS"""
        mock_fs = mocker.patch("fsspec.filesystem")
        credentials = {"key": "value"}

        Sas7bdatDataset(filepath=FILENAME, credentials=credentials)

        mock_fs.assert_called_once_with("file", auto_mkdir=True, **credentials)

    def test_exists(self, sas_data_set):
        """Test `exists` method invocation for both existing and
        nonexistent data set."""
        assert sas_data_set.exists()

    @pytest.mark.parametrize(
        "load_args", [{"k1": "v1", "index": "value"}], indirect=True
    )
    def test_load_extra_params(self, sas_data_set, load_args):
        """Test overriding the default load arguments."""
        for key, value in load_args.items():
            assert sas_data_set._load_args[key] == value

    @pytest.mark.parametrize(
        "save_args", [{"k1": "v1", "index": "value"}], indirect=True
    )
    def test_save_extra_params(self, sas_data_set, save_args):
        """Test overriding the default save arguments."""
        for key, value in save_args.items():
            assert sas_data_set._save_args[key] == value

    @pytest.mark.parametrize(
        "load_args,save_args",
        [
            ({"storage_options": {"a": "b"}}, {}),
            ({}, {"storage_options": {"a": "b"}}),
            ({"storage_options": {"a": "b"}}, {"storage_options": {"x": "y"}}),
        ],
    )
    def test_storage_options_dropped(self, load_args, save_args, caplog, tmp_path):
        filepath = str(tmp_path / "test.csv")

        ds = Sas7bdatDataset(
            filepath=filepath, load_args=load_args, save_args=save_args
        )

        records = [r for r in caplog.records if r.levelname == "WARNING"]
        expected_log_message = (
            f"Dropping 'storage_options' for {filepath}, "
            f"please specify them under 'fs_args' or 'credentials'."
        )
        assert records[0].getMessage() == expected_log_message
        assert "storage_options" not in ds._save_args
        assert "storage_options" not in ds._load_args

    def test_load_missing_file(self):
        """Check the error when trying to load missing file."""
        pattern = r"Failed while loading data from data set Sas7bdatDataset\(.*\)"
        with pytest.raises(DataSetError, match=pattern):
            data_set = Sas7bdatDataset("/unknown.sas7bdat")
            data_set.load()

    @pytest.mark.parametrize(
        "filepath,instance_type,load_path",
        [
            ("s3://bucket/file.sas7bdat", S3FileSystem, "s3://bucket/file.sas7bdat"),
            ("file:///tmp/test.sas7bdat", LocalFileSystem, "/tmp/test.sas7bdat"),
            ("/tmp/test.sas7bdat", LocalFileSystem, "/tmp/test.sas7bdat"),
            ("gcs://bucket/file.sas7bdat", GCSFileSystem, "gcs://bucket/file.sas7bdat"),
            (
                "https://example.com/file.sas7bdat",
                HTTPFileSystem,
                "https://example.com/file.sas7bdat",
            ),
        ],
    )
    def test_protocol_usage(self, filepath, instance_type, load_path, mocker):
        data_set = Sas7bdatDataset(filepath=filepath)
        assert isinstance(data_set._fs, instance_type)

        path = filepath.split(PROTOCOL_DELIMITER, 1)[-1]

        assert str(data_set._filepath) == path
        assert isinstance(data_set._filepath, PurePosixPath)

        mocker.patch.object(data_set._fs, "isdir", return_value=False)
        mock_pandas_call = mocker.patch("pandas.read_sas")
        data_set.load()
        assert mock_pandas_call.call_count == 1
        assert mock_pandas_call.call_args_list[0][0][0] == load_path

    @pytest.mark.parametrize(
        "protocol,path", [("https://", "example.com/"), ("s3://", "bucket/")]
    )
    def test_catalog_release(self, protocol, path, mocker):
        filepath = protocol + path + FILENAME
        fs_mock = mocker.patch("fsspec.filesystem").return_value
        data_set = Sas7bdatDataset(filepath=filepath)
        data_set.release()
        if protocol != "https://":
            filepath = path + FILENAME
        fs_mock.invalidate_cache.assert_called_once_with(filepath)

    def test_read_from_non_local_dir(self, mocker):
        mock_pandas_call = mocker.patch("pandas.read_sas")

        data_set = Sas7bdatDataset(filepath="s3://bucket/dir")

        data_set.load()
        assert mock_pandas_call.call_count == 1

    def test_read_from_file(self, mocker):
        mock_pandas_call = mocker.patch("pandas.read_sas")

        data_set = Sas7bdatDataset(filepath="/tmp/test.sasb7bat")

        data_set.load()
        assert mock_pandas_call.call_count == 1


class TestSas7bdatDatasetVersioned:
    def test_version_str_repr(self, load_version, save_version):
        """Test that version is in string representation of the class instance
        when applicable."""
        ds = Sas7bdatDataset(filepath=FILENAME)
        ds_versioned = Sas7bdatDataset(
            filepath=FILENAME, version=Version(load_version, save_version)
        )
        assert FILENAME in str(ds)
        assert "version" not in str(ds)

        assert FILENAME in str(ds_versioned)
        ver_str = f"version=Version(load={load_version}, save='{save_version}')"
        assert ver_str in str(ds_versioned)
        assert "Sas7bdatDataset" in str(ds_versioned)
        assert "Sas7bdatDataset" in str(ds)
        assert "protocol" in str(ds_versioned)
        assert "protocol" in str(ds)

    def test_no_versions(self, versioned_sas_data_set):
        """Check the error if no versions are available for load."""
        pattern = r"Did not find any versions for Sas7bdatDataset\(.+\)"
        with pytest.raises(DataSetError, match=pattern):
            versioned_sas_data_set.load()

    def test_http_filesystem_no_versioning(self):
        pattern = "Versioning is not supported for HTTP protocols."

        with pytest.raises(DataSetError, match=pattern):
            Sas7bdatDataset(
                filepath="https://example.com/test.sasb7dat",
                version=Version(None, None),
            )

    def test_save_error(self, sas_data_set, dummy_dataframe):
        """Check the error when trying to save to the data set"""
        pattern = r"'save' is not supported on Sas7bdatDataset"
        with pytest.raises(DataSetError, match=pattern):
            sas_data_set.save(dummy_dataframe)
