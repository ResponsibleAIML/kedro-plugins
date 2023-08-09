import base64
import io

import certifi
import pyarrow as pa
import pytest
from kedro.io.core import DataSetError
from pandas import DataFrame
from pyarrow import flight
from pyarrow.flight import FlightServerBase, ServerMiddlewareFactory
from pyarrow.tests.test_flight import (
    GetInfoFlightServer,
    HeaderAuthServerMiddleware,
    HttpBasicServerAuthHandler,
    NoopAuthHandler,
)

from kedro_datasets.arrowflight import ArrowFlightDataSet
from kedro_datasets.arrowflight.flight_dataset import (
    ClientMiddleware,
    ClientMiddlewareFactory,
    HttpClientAuthHandler,
    process_credentials,
)


@pytest.fixture
def sql_path_sql(tmp_path):
    return (tmp_path / "test.sql").as_posix()


@pytest.fixture
def default_flight_host():
    return "localhost"


def simple_ints_table():
    data = [pa.array([-10, -5, 0, 5, 10])]
    return pa.Table.from_arrays(data, names=["some_ints"])


def simple_dicts_table():
    dict_values = pa.array(["foo", "baz", "quux"], type=pa.utf8())
    data = [
        pa.chunked_array(
            [
                pa.DictionaryArray.from_arrays([1, 0, None], dict_values),
                pa.DictionaryArray.from_arrays([2, 1], dict_values),
            ]
        )
    ]
    return pa.Table.from_arrays(data, names=["some_dicts"])


def multiple_column_table():
    return pa.Table.from_arrays(
        [pa.array(["foo", "bar", "baz", "qux"]), pa.array([1, 2, 3, 4])],
        names=["a", "b"],
    )


basic_auth_handler = HttpBasicServerAuthHandler(
    creds={
        b"username": b"password",
    }
)


no_op_auth_handler = NoopAuthHandler()


class HeaderAuthServerMiddlewareFactory(ServerMiddlewareFactory):
    """Validates incoming username and password."""

    def start_call(self, _, headers):
        auth_header = headers.get("authorization")

        values = auth_header[0].split(" ")
        token = ""
        error_message = "Invalid credentials"

        if values[0] == "Basic":
            decoded = base64.b64decode(values[1])
            pair = decoded.decode("utf-8").split(":")
            if not (pair[0] == "username" and pair[1] == "password"):
                raise flight.FlightUnauthenticatedError(error_message)
            token = "token1234"
        elif values[0] == "Bearer":
            token = values[1]
            if not token == "token1234":
                raise flight.FlightUnauthenticatedError(error_message)
        else:
            raise flight.FlightUnauthenticatedError(f"{error_message}{values}")

        return HeaderAuthServerMiddleware(token)


class ConstantFlightServer(FlightServerBase):
    """A Flight server that always returns the same data.

    See ARROW-4796: this server implementation will segfault if Flight
    does not properly hold a reference to the Table object.
    """

    CRITERIA = b"the expected criteria"

    def __init__(self, location=None, options=None, **kwargs):
        super().__init__(location, **kwargs)
        # Ticket -> Table
        self.table_factories = {
            b"ints": simple_ints_table,
            b"dicts": simple_dicts_table,
            b"multi": multiple_column_table,
        }
        self.options = options

    def list_flights(self, _, criteria):
        if criteria == self.CRITERIA:
            yield flight.FlightInfo(
                pa.schema([]), flight.FlightDescriptor.for_path("/foo"), [], -1, -1
            )

    def do_get(self, _, ticket):
        # Return a fresh table, so that Flight is the only one keeping a
        # reference.
        table = self.table_factories[ticket.ticket]()
        return flight.RecordBatchStream(table, options=self.options)

    def get_flight_info(self, _, descriptor):
        return flight.FlightInfo(
            pa.schema([("a", pa.int32())]),
            descriptor,
            [
                flight.FlightEndpoint(b"ints", ["grpc://test"]),
                flight.FlightEndpoint(
                    b"",
                    [flight.Location.for_grpc_tcp("localhost", 5005)],
                ),
            ],
            -1,
            -1,
        )

    def do_put(self, context, descriptor, reader, writer):
        _ = context
        _ = descriptor
        _ = writer
        for _ in reader:
            pass


class HeaderAuthFlightServer(ConstantFlightServer):
    """A Flight server that tests with basic token authentication."""

    def do_action(self, context, _):
        middleware = context.get_middleware("auth")
        if middleware:
            auth_header = "authorization"
            values = auth_header.split(" ")
            return [values[1].encode("utf-8")]
        raise flight.FlightUnauthenticatedError("No token auth middleware found.")


class TestProcessCon:
    def test_credentials_con_with_username_and_password(self, default_flight_host):
        credentials = {"con": f"username:password@{default_flight_host}:port"}
        result = process_credentials(credentials)
        assert result["protocol"] == "grpc+tcp"
        assert result["hostname"] == default_flight_host
        assert result["username"] == "username"
        assert result["password"] == "password"

    def test_credentials_con_with_username_and_password_and_protocal(
        self, default_flight_host
    ):
        credentials = {"con": f"grpc://username:password@{default_flight_host}:port"}
        result = process_credentials(credentials)
        assert result["protocol"] == "grpc"
        assert result["hostname"] == default_flight_host
        assert result["username"] == "username"
        assert result["password"] == "password"

    def test_credentials_con_with_username_and_password_and_tls(
        self, default_flight_host
    ):
        credentials = {
            "con": f"username:password@{default_flight_host}:port",
            "tls": True,
        }
        result = process_credentials(credentials)
        assert result["protocol"] == "grpc+tls"
        assert result["hostname"] == default_flight_host
        assert result["username"] == "username"
        assert result["password"] == "password"

    def test_credentials_con_with_username_and_password_provided_explicitly(
        self, default_flight_host
    ):
        credentials = {
            "con": default_flight_host,
            "username": "username",
            "password": "password",
        }

        result = process_credentials(credentials)
        assert result["protocol"] == "grpc+tcp"
        assert result["hostname"] == default_flight_host
        assert result["username"] == credentials["username"]
        assert result["password"] == credentials["password"]

    def test_credentials_con_with_credentials_provided_explicitly_duplicate(
        self, default_flight_host
    ):
        credentials = {
            "con": f"username:password@{default_flight_host}:port",
            "username": "username",
            "password": "password",
        }
        # pylint: disable=C0301
        pattern = "Arrow URI must not include username and password if they were supplied explicitly"
        with pytest.raises(ValueError, match=pattern):
            process_credentials(credentials)

    def test_credentials_con_without_username_and_password(self, default_flight_host):
        result = process_credentials({"con": default_flight_host})
        assert result["hostname"] == default_flight_host
        assert result["username"] is None
        assert result["password"] is None

    def test_credentials_no_dictionary(self):
        pattern = "Credentials dict can not be empty"
        with pytest.raises(ValueError, match=pattern):
            process_credentials(None)

    def test_credentials_no_uri(self):
        pattern = "Arrow URI can not be empty"
        with pytest.raises(ValueError, match=pattern):
            process_credentials({"con": None})
        with pytest.raises(ValueError, match=pattern):
            process_credentials({"con": ""})


class TestHttpClientAuthHandler:
    def test_init(self):
        handler = HttpClientAuthHandler("username", "password")
        basic_auth = handler.basic_auth
        token = handler.token
        assert basic_auth
        assert token is None

    def test_authenticate(self):
        handler = HttpClientAuthHandler("username", "password")
        outgoing = io.BytesIO()
        incoming = io.BytesIO()
        handler.authenticate(outgoing, incoming)
        outgoing = outgoing.getvalue()
        incoming = incoming.getvalue()
        token = handler.token
        assert token == b""

    def test_get_token(self):
        handler = HttpClientAuthHandler("username", "password")
        handler.token = b"token"
        token = handler.get_token()
        assert token


class TestClientMiddleware:
    def test_received_headers(self):
        factory = ClientMiddlewareFactory()
        middleware = ClientMiddleware(factory)
        headers = {"authorization": ["Bearer my-token"]}
        middleware.received_headers(headers)
        assert middleware.factory.call_credential == [
            b"authorization",
            b"Bearer my-token",
        ]

    def test_received_headers_no_authorization(self):
        factory = ClientMiddlewareFactory()
        middleware = ClientMiddleware(factory)
        headers = {}
        middleware.received_headers(headers)
        assert middleware.factory.call_credential == []


class TestClientMiddlewareFactory:
    def test_start_call(self):
        factory = ClientMiddlewareFactory()
        factory.set_call_credential([b"credential1", b"credential2"])
        middleware = factory.start_call(None)
        assert middleware.factory.call_credential == [b"credential1", b"credential2"]

    def test_set_call_credential(self):
        factory = ClientMiddlewareFactory()
        factory.set_call_credential([b"credential1", b"credential2"])
        assert factory.call_credential == [b"credential1", b"credential2"]


class TestArrowFlightDataSet:
    def test_init_with_sql(self):
        with ConstantFlightServer() as server:
            credentials = {"con": f"username:password@localhost:{server.port}"}
            dataset = ArrowFlightDataSet(
                sql="SELECT * FROM users", credentials=credentials
            )
            assert dataset._load_args["sql"] == "SELECT * FROM users"

    def test_init_with_sql_path(self):
        with ConstantFlightServer() as server:
            credentials = {"con": f"username:password@localhost:{server.port}"}
            sql_path = "/path/to/file.sql"
            dataset = ArrowFlightDataSet(sql_path=sql_path, credentials=credentials)
            assert dataset._load_args["sql_path"] == sql_path

    def test_load(self):
        with ConstantFlightServer() as server:
            credentials = {"con": f"localhost:{server.port}"}
            dataset = ArrowFlightDataSet(
                sql="SELECT * FROM users", credentials=credentials
            )
            df = dataset._load()
            assert df.shape[0] > 0

    def test_load_sql_path(self, tmp_path):
        with ConstantFlightServer() as server:
            credentials = {"con": f"localhost:{server.port}"}
            sql_path = tmp_path / "file.sql"
            sql_path.write_text("b'ints'")
            dataset = ArrowFlightDataSet(sql_path=sql_path, credentials=credentials)
            df = dataset._load()
            assert df.shape[0] > 0

    def test_load_authenticate(self):
        with ConstantFlightServer(auth_handler=basic_auth_handler) as server:
            credentials = {"con": f"username:password@localhost:{server.port}"}
            dataset = ArrowFlightDataSet(
                sql="SELECT * FROM users", credentials=credentials
            )
            df = dataset._load()
            assert df.shape[0] > 0

    def test_load_basic_authenticate(self):
        with HeaderAuthFlightServer(
            auth_handler=no_op_auth_handler,
            middleware={"auth": HeaderAuthServerMiddlewareFactory()},
        ) as server:
            credentials = {"con": f"username:password@localhost:{server.port}"}
            dataset = ArrowFlightDataSet(
                sql="SELECT * FROM users", credentials=credentials
            )
            df = dataset._load()
            assert df.shape[0] > 0

    #:TODO
    def test_load_tls_certs(self):
        certs_path = certifi.where()
        with ConstantFlightServer() as server:
            credentials = {
                "con": f"localhost:{server.port}",
                "tls": True,
                "certs": certs_path,
                "disable_server_verification": True,
            }
            try:
                dataset = ArrowFlightDataSet(
                    sql="SELECT * FROM users",
                    credentials=credentials,
                )
                df = dataset._load()
                assert df.shape[0] > 0
            except flight.FlightUnavailableError:
                pytest.skip("disable_server_verification feature is not available")

    def test_exists_not_supported(self):
        with ConstantFlightServer() as server:
            credentials = {"con": f"localhost:{server.port}"}
            pattern = "'exists' is not supported on ArrowFlightDataSet"
            dataset = ArrowFlightDataSet(sql=b"ints", credentials=credentials)
            with pytest.raises(DataSetError, match=pattern):
                dataset._exists()

    def test_save_not_supported(self):
        with ConstantFlightServer() as server:
            credentials = {"con": f"localhost:{server.port}"}
            pattern = "'save' is not supported on ArrowFlightDataSet"
            dataset = ArrowFlightDataSet(sql=b"ints", credentials=credentials)
            with pytest.raises(DataSetError, match=pattern):
                dataset._save(DataFrame())

    def test_sql_and_sql_path_is_invalid(self):
        with ConstantFlightServer() as server:
            credentials = {"con": f"username:password@localhost:{server.port}"}
            pattern = "'sql' and 'sql_path' arguments cannot both be provided"
            with pytest.raises(DataSetError, match=pattern):
                ArrowFlightDataSet(
                    sql="select * from dual",
                    sql_path="test.sql",
                    credentials=credentials,
                )

    def test_sql_and_sql_path_are_empty(self):
        with ConstantFlightServer() as server:
            credentials = {"con": f"username:password@localhost:{server.port}"}
            pattern = "'sql' and 'sql_path' arguments cannot both be empty."
            with pytest.raises(DataSetError, match=pattern):
                ArrowFlightDataSet(credentials=credentials)

    def test_con_is_empty(self):
        pattern = "'con' argument cannot be empty."
        with pytest.raises(DataSetError, match=pattern):
            ArrowFlightDataSet(sql="select * from dual")

    def test_load_certs(self):
        certs_path = certifi.where()
        with ConstantFlightServer() as server:
            credentials = {
                "con": f"username:password@localhost:{server.port}",
                "certs": certs_path,
                "tls": True,
            }
            dataset = ArrowFlightDataSet(sql=b"ints", credentials=credentials)
            assert dataset._certs

    def test_tls_no_certs(self):
        with ConstantFlightServer() as server:
            credentials = {
                "con": f"username:password@localhost:{server.port}",
                "tls": True,
            }
            pattern = (
                "Trusted certificates must be provided to establish a TLS connection"
            )
            with pytest.raises(ValueError, match=pattern):
                ArrowFlightDataSet(sql=b"ints", credentials=credentials)

    def test_describe(self):
        with GetInfoFlightServer() as server:
            credentials = {"con": f"localhost:{server.port}"}
            dataset = ArrowFlightDataSet(sql="sql", credentials=credentials)
            info = dataset._describe()
            assert info == {
                "load_args": {
                    "connect_timeout": None,
                    "return_pandas": True,
                },
                "sql": "sql",
                "sql_path": None,
            }

    def test_client_close(self):
        with ConstantFlightServer() as server:
            credentials = {"con": f"localhost:{server.port}"}
            dataset = ArrowFlightDataSet(
                sql="SELECT * FROM users", credentials=credentials
            )
            dataset.close()
            dataset._load()
            dataset.close()
