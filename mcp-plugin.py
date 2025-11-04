import os
import sys

if sys.version_info < (3, 11):
    raise RuntimeError("Python 3.11 or higher is required for the MCP plugin")

import json
import struct
import threading
import http.server
from urllib.parse import urlparse
from typing import Any, Callable, get_type_hints, TypedDict, Optional, Annotated, TypeVar, Generic, NotRequired

# Keystone Engine imports for enhanced assembly support
try:
    from keystone import *
    KEYSTONE_AVAILABLE = True
except ImportError:
    KEYSTONE_AVAILABLE = False


class JSONRPCError(Exception):
    def __init__(self, code: int, message: str, data: Any = None):
        self.code = code
        self.message = message
        self.data = data

class RPCRegistry:
    def __init__(self):
        self.methods: dict[str, Callable] = {}
        self.unsafe: set[str] = set()

    def register(self, func: Callable) -> Callable:
        self.methods[func.__name__] = func
        return func

    def mark_unsafe(self, func: Callable) -> Callable:
        self.unsafe.add(func.__name__)
        return func

    def dispatch(self, method: str, params: Any) -> Any:
        if method not in self.methods:
            raise JSONRPCError(-32601, f"Method '{method}' not found")

        func = self.methods[method]
        hints = get_type_hints(func)

        # Remove return annotation if present
        hints.pop("return", None)

        if isinstance(params, list):
            if len(params) != len(hints):
                raise JSONRPCError(-32602, f"Invalid params: expected {len(hints)} arguments, got {len(params)}")

            # Validate and convert parameters
            converted_params = []
            for value, (param_name, expected_type) in zip(params, hints.items()):
                try:
                    if not isinstance(value, expected_type):
                        value = expected_type(value)
                    converted_params.append(value)
                except (ValueError, TypeError):
                    raise JSONRPCError(-32602, f"Invalid type for parameter '{param_name}': expected {expected_type.__name__}")

            return func(*converted_params)
        elif isinstance(params, dict):
            if set(params.keys()) != set(hints.keys()):
                raise JSONRPCError(-32602, f"Invalid params: expected {list(hints.keys())}")

            # Validate and convert parameters
            converted_params = {}
            for param_name, expected_type in hints.items():
                value = params.get(param_name)
                try:
                    if not isinstance(value, expected_type):
                        value = expected_type(value)
                    converted_params[param_name] = value
                except (ValueError, TypeError):
                    raise JSONRPCError(-32602, f"Invalid type for parameter '{param_name}': expected {expected_type.__name__}")

            return func(**converted_params)
        else:
            raise JSONRPCError(-32600, "Invalid Request: params must be array or object")

rpc_registry = RPCRegistry()

def jsonrpc(func: Callable) -> Callable:
    """Decorator to register a function as a JSON-RPC method"""
    global rpc_registry
    return rpc_registry.register(func)

def unsafe(func: Callable) -> Callable:
    """Decorator to register mark a function as unsafe"""
    return rpc_registry.mark_unsafe(func)

class JSONRPCRequestHandler(http.server.BaseHTTPRequestHandler):
    def send_jsonrpc_error(self, code: int, message: str, id: Any = None):
        response = {
            "jsonrpc": "2.0",
            "error": {
                "code": code,
                "message": message
            }
        }
        if id is not None:
            response["id"] = id
        response_body = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(response_body))
        self.end_headers()
        self.wfile.write(response_body)

    def do_POST(self):
        global rpc_registry

        parsed_path = urlparse(self.path)
        if parsed_path.path != "/mcp":
            self.send_jsonrpc_error(-32098, "Invalid endpoint", None)
            return

        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            self.send_jsonrpc_error(-32700, "Parse error: missing request body", None)
            return

        request_body = self.rfile.read(content_length)
        try:
            request = json.loads(request_body)
        except json.JSONDecodeError:
            self.send_jsonrpc_error(-32700, "Parse error: invalid JSON", None)
            return

        # Prepare the response
        response = {
            "jsonrpc": "2.0"
        }
        if request.get("id") is not None:
            response["id"] = request.get("id")

        try:
            # Basic JSON-RPC validation
            if not isinstance(request, dict):
                raise JSONRPCError(-32600, "Invalid Request")
            if request.get("jsonrpc") != "2.0":
                raise JSONRPCError(-32600, "Invalid JSON-RPC version")
            if "method" not in request:
                raise JSONRPCError(-32600, "Method not specified")

            # Dispatch the method
            result = rpc_registry.dispatch(request["method"], request.get("params", []))
            response["result"] = result

        except JSONRPCError as e:
            response["error"] = {
                "code": e.code,
                "message": e.message
            }
            if e.data is not None:
                response["error"]["data"] = e.data
        except IDAError as e:
            response["error"] = {
                "code": -32000,
                "message": e.message,
            }
        except Exception as e:
            traceback.print_exc()
            response["error"] = {
                "code": -32603,
                "message": "Internal error (please report a bug)",
                "data": traceback.format_exc(),
            }

        try:
            response_body = json.dumps(response).encode("utf-8")
        except Exception as e:
            traceback.print_exc()
            response_body = json.dumps({
                "error": {
                    "code": -32603,
                    "message": "Internal error (please report a bug)",
                    "data": traceback.format_exc(),
                }
            }).encode("utf-8")

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(response_body))
        self.end_headers()
        self.wfile.write(response_body)

    def log_message(self, format, *args):
        # Suppress logging
        pass

class MCPHTTPServer(http.server.HTTPServer):
    allow_reuse_address = False

class Server:
    HOST = "localhost"
    PORT = 13337

    def __init__(self):
        self.server = None
        self.server_thread = None
        self.running = False

    def start(self):
        if self.running:
            print("[MCP] Server is already running")
            return

        self.server_thread = threading.Thread(target=self._run_server, daemon=True)
        self.running = True
        self.server_thread.start()

    def stop(self):
        if not self.running:
            return

        self.running = False
        if self.server:
            self.server.shutdown()
            self.server.server_close()
        if self.server_thread:
            self.server_thread.join()
            self.server = None
        print("[MCP] Server stopped")

    def _run_server(self):
        try:
            # Create server in the thread to handle binding
            self.server = MCPHTTPServer((Server.HOST, Server.PORT), JSONRPCRequestHandler)
            print(f"[MCP] Server started at http://{Server.HOST}:{Server.PORT}")
            self.server.serve_forever()
        except OSError as e:
            if e.errno == 98 or e.errno == 10048:  # Port already in use (Linux/Windows)
                print("[MCP] Error: Port 13337 is already in use")
            else:
                print(f"[MCP] Server error: {e}")
            self.running = False
        except Exception as e:
            print(f"[MCP] Server error: {e}")
        finally:
            self.running = False

# A module that helps with writing thread safe ida code.
# Based on:
# https://web.archive.org/web/20160305190440/http://www.williballenthin.com/blog/2015/09/04/idapython-synchronization-decorator/
import logging
import queue
import traceback
import functools

import ida_hexrays
import ida_kernwin
import ida_funcs
import ida_lines
import ida_idaapi
import idc
import idaapi
import idautils
import ida_nalt
import ida_bytes
import ida_typeinf
import ida_xref
import ida_name
import ida_ida

# Architecture mapping for Keystone Engine - Updated for IDA Pro 9.1 compatibility

def get_keystone_architecture():
    """Get the appropriate Keystone architecture and mode for current IDA session - IDA 9.1 compatible"""
    if not KEYSTONE_AVAILABLE:
        return None, None
    
    # Get processor name using IDA 9.1 compatible method
    try:
        procname = ida_ida.inf_get_procname().lower()
        print(f"DEBUG: Got procname from inf_get_procname: '{procname}'")
    except Exception as e:
        # Fallback for different IDA versions
        print(f"DEBUG: inf_get_procname failed: {e}")
        try:
            procname = ida_idaapi.get_inf_structure().procname.lower()
            print(f"DEBUG: Got procname from get_inf_structure: '{procname}'")
        except Exception as e2:
            print(f"DEBUG: get_inf_structure also failed: {e2}")
            return None, None
    
    # Architecture detection based on processor name
    print(f"DEBUG: Checking procname '{procname}' for architecture detection")
    
    # Check if it's 64-bit first to properly distinguish ARM64 from ARM32
    is_64bit = ida_ida.inf_is_64bit()
    print(f"DEBUG: IDA inf_is_64bit() = {is_64bit}")
    
    if 'arm' in procname:
        if is_64bit or 'arm64' in procname or 'aarch64' in procname:
            print("DEBUG: Detected ARM64 architecture")
            return KS_ARCH_ARM64, KS_MODE_LITTLE_ENDIAN
        else:
            print("DEBUG: Detected ARM32 architecture")
            return KS_ARCH_ARM, KS_MODE_ARM
    elif procname in ['metapc', 'pc'] or '386' in procname or 'x86' in procname:
        if ida_ida.inf_is_64bit():
            return KS_ARCH_X86, KS_MODE_64
        else:
            return KS_ARCH_X86, KS_MODE_32
    elif 'mips' in procname:
        return KS_ARCH_MIPS, KS_MODE_MIPS64 if ida_ida.inf_is_64bit() else KS_MODE_MIPS32
    elif 'ppc' in procname or 'powerpc' in procname:
        return KS_ARCH_PPC, KS_MODE_PPC64 if ida_ida.inf_is_64bit() else KS_MODE_PPC32
    elif 'sparc' in procname:
        return KS_ARCH_SPARC, KS_MODE_SPARC64 if ida_ida.inf_is_64bit() else KS_MODE_SPARC32
    
    # Unknown architecture
    return None, None

class IDAError(Exception):
    def __init__(self, message: str):
        super().__init__(message)

    @property
    def message(self) -> str:
        return self.args[0]

class IDASyncError(Exception):
    pass

class DecompilerLicenseError(IDAError):
    pass

# Important note: Always make sure the return value from your function f is a
# copy of the data you have gotten from IDA, and not the original data.
#
# Example:
# --------
#
# Do this:
#
#   @idaread
#   def ts_Functions():
#       return list(idautils.Functions())
#
# Don't do this:
#
#   @idaread
#   def ts_Functions():
#       return idautils.Functions()
#

logger = logging.getLogger(__name__)

# Enum for safety modes. Higher means safer:
class IDASafety:
    SAFE_NONE = ida_kernwin.MFF_FAST
    SAFE_READ = ida_kernwin.MFF_READ
    SAFE_WRITE = ida_kernwin.MFF_WRITE

call_stack = queue.LifoQueue()

def sync_wrapper(ff, safety_mode: IDASafety):
    """
    Call a function ff with a specific IDA safety_mode.
    """
    #logger.debug('sync_wrapper: {}, {}'.format(ff.__name__, safety_mode))

    if safety_mode not in [IDASafety.SAFE_READ, IDASafety.SAFE_WRITE]:
        error_str = 'Invalid safety mode {} over function {}'\
                .format(safety_mode, ff.__name__)
        logger.error(error_str)
        raise IDASyncError(error_str)

    # No safety level is set up:
    res_container = queue.Queue()

    def runned():
        #logger.debug('Inside runned')

        # Make sure that we are not already inside a sync_wrapper:
        if not call_stack.empty():
            last_func_name = call_stack.get()
            error_str = ('Call stack is not empty while calling the '
                'function {} from {}').format(ff.__name__, last_func_name)
            #logger.error(error_str)
            raise IDASyncError(error_str)

        call_stack.put((ff.__name__))
        try:
            res_container.put(ff())
        except Exception as x:
            res_container.put(x)
        finally:
            call_stack.get()
            #logger.debug('Finished runned')

    ret_val = idaapi.execute_sync(runned, safety_mode)
    res = res_container.get()
    if isinstance(res, Exception):
        raise res
    return res

def idawrite(f):
    """
    decorator for marking a function as modifying the IDB.
    schedules a request to be made in the main IDA loop to avoid IDB corruption.
    """
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        ff = functools.partial(f, *args, **kwargs)
        ff.__name__ = f.__name__
        return sync_wrapper(ff, idaapi.MFF_WRITE)
    return wrapper

def idaread(f):
    """
    decorator for marking a function as reading from the IDB.
    schedules a request to be made in the main IDA loop to avoid
      inconsistent results.
    MFF_READ constant via: http://www.openrce.org/forums/posts/1827
    """
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        ff = functools.partial(f, *args, **kwargs)
        ff.__name__ = f.__name__
        return sync_wrapper(ff, idaapi.MFF_READ)
    return wrapper

def is_window_active():
    """Returns whether IDA is currently active"""
    try:
        from PyQt5.QtWidgets import QApplication
    except ImportError:
        return False

    app = QApplication.instance()
    if app is None:
        return False

    for widget in app.topLevelWidgets():
        if widget.isActiveWindow():
            return True
    return False

class Metadata(TypedDict):
    path: str
    module: str
    base: str
    size: str
    md5: str
    sha256: str
    crc32: str
    filesize: str

def get_image_size() -> int:
    try:
        # https://www.hex-rays.com/products/ida/support/sdkdoc/structidainfo.html
        info = idaapi.get_inf_structure()
        omin_ea = info.omin_ea
        omax_ea = info.omax_ea
    except AttributeError:
        import ida_ida
        omin_ea = ida_ida.inf_get_omin_ea()
        omax_ea = ida_ida.inf_get_omax_ea()
    # Bad heuristic for image size (bad if the relocations are the last section)
    image_size = omax_ea - omin_ea
    # Try to extract it from the PE header
    header = idautils.peutils_t().header()
    if header and header[:4] == b"PE\0\0":
        image_size = struct.unpack("<I", header[0x50:0x54])[0]
    return image_size

@jsonrpc
@idaread
def get_metadata() -> Metadata:
    """
    Retrieve comprehensive metadata about the currently loaded IDA database.
    
    Returns detailed information about the binary file including file paths, 
    base addresses, size, and cryptographic hashes. This is a read-only operation
    that provides essential context for reverse engineering analysis.
    
    Use this as a starting point to understand the loaded binary's characteristics.
    """
    # Fat Mach-O binaries can return a None hash:
    # https://github.com/mrexodia/ida-pro-mcp/issues/26
    def hash(f):
        try:
            return f().hex()
        except:
            return None

    return Metadata(path=idaapi.get_input_file_path(),
                    module=idaapi.get_root_filename(),
                    base=hex(idaapi.get_imagebase()),
                    size=hex(get_image_size()),
                    md5=hash(ida_nalt.retrieve_input_file_md5),
                    sha256=hash(ida_nalt.retrieve_input_file_sha256),
                    crc32=hex(ida_nalt.retrieve_input_file_crc32()),
                    filesize=hex(ida_nalt.retrieve_input_file_size()))

def get_prototype(fn: ida_funcs.func_t) -> Optional[str]:
    try:
        prototype: ida_typeinf.tinfo_t = fn.get_prototype()
        if prototype is not None:
            return str(prototype)
        else:
            return None
    except AttributeError:
        try:
            return idc.get_type(fn.start_ea)
        except:
            tif = ida_typeinf.tinfo_t()
            if ida_nalt.get_tinfo(tif, fn.start_ea):
                return str(tif)
            return None
    except Exception as e:
        print(f"Error getting function prototype: {e}")
        return None

class Function(TypedDict):
    address: str
    name: str
    size: str

def parse_address(address: str) -> int:
    try:
        return int(address, 0)
    except ValueError:
        for ch in address:
            if ch not in "0123456789abcdefABCDEF":
                raise IDAError(f"Failed to parse address: {address}")
        raise IDAError(f"Failed to parse address (missing 0x prefix): {address}")

def get_function(address: int, *, raise_error=True) -> Function:
    fn = idaapi.get_func(address)
    if fn is None:
        if raise_error:
            raise IDAError(f"No function found at address {hex(address)}")
        return None

    try:
        name = fn.get_name()
    except AttributeError:
        name = ida_funcs.get_func_name(fn.start_ea)

    return Function(address=hex(address), name=name, size=hex(fn.end_ea - fn.start_ea))

DEMANGLED_TO_EA = {}

def create_demangled_to_ea_map():
    for ea in idautils.Functions():
        # Get the function name and demangle it
        # MNG_NODEFINIT inhibits everything except the main name
        # where default demangling adds the function signature
        # and decorators (if any)
        demangled = idaapi.demangle_name(
            idc.get_name(ea, 0), idaapi.MNG_NODEFINIT)
        if demangled:
            DEMANGLED_TO_EA[demangled] = ea


def get_type_by_name(type_name: str) -> ida_typeinf.tinfo_t:
    # 8-bit integers
    if type_name in ('int8', '__int8', 'int8_t', 'char', 'signed char'):
        return ida_typeinf.tinfo_t(ida_typeinf.BTF_INT8)
    elif type_name in ('uint8', '__uint8', 'uint8_t', 'unsigned char', 'byte', 'BYTE'):
        return ida_typeinf.tinfo_t(ida_typeinf.BTF_UINT8)

    # 16-bit integers
    elif type_name in ('int16', '__int16', 'int16_t', 'short', 'short int', 'signed short', 'signed short int'):
        return ida_typeinf.tinfo_t(ida_typeinf.BTF_INT16)
    elif type_name in ('uint16', '__uint16', 'uint16_t', 'unsigned short', 'unsigned short int', 'word', 'WORD'):
        return ida_typeinf.tinfo_t(ida_typeinf.BTF_UINT16)

    # 32-bit integers
    elif type_name in ('int32', '__int32', 'int32_t', 'int', 'signed int', 'long', 'long int', 'signed long', 'signed long int'):
        return ida_typeinf.tinfo_t(ida_typeinf.BTF_INT32)
    elif type_name in ('uint32', '__uint32', 'uint32_t', 'unsigned int', 'unsigned long', 'unsigned long int', 'dword', 'DWORD'):
        return ida_typeinf.tinfo_t(ida_typeinf.BTF_UINT32)

    # 64-bit integers
    elif type_name in ('int64', '__int64', 'int64_t', 'long long', 'long long int', 'signed long long', 'signed long long int'):
        return ida_typeinf.tinfo_t(ida_typeinf.BTF_INT64)
    elif type_name in ('uint64', '__uint64', 'uint64_t', 'unsigned int64', 'unsigned long long', 'unsigned long long int', 'qword', 'QWORD'):
        return ida_typeinf.tinfo_t(ida_typeinf.BTF_UINT64)

    # 128-bit integers
    elif type_name in ('int128', '__int128', 'int128_t', '__int128_t'):
        return ida_typeinf.tinfo_t(ida_typeinf.BTF_INT128)
    elif type_name in ('uint128', '__uint128', 'uint128_t', '__uint128_t', 'unsigned int128'):
        return ida_typeinf.tinfo_t(ida_typeinf.BTF_UINT128)

    # Floating point types
    elif type_name in ('float', ):
        return ida_typeinf.tinfo_t(ida_typeinf.BTF_FLOAT)
    elif type_name in ('double', ):
        return ida_typeinf.tinfo_t(ida_typeinf.BTF_DOUBLE)
    elif type_name in ('long double', 'ldouble'):
        return ida_typeinf.tinfo_t(ida_typeinf.BTF_LDOUBLE)

    # Boolean type
    elif type_name in ('bool', '_Bool', 'boolean'):
        return ida_typeinf.tinfo_t(ida_typeinf.BTF_BOOL)

    # Void type
    elif type_name in ('void', ):
        return ida_typeinf.tinfo_t(ida_typeinf.BTF_VOID)

    # If not a standard type, try to get a named type
    tif = ida_typeinf.tinfo_t()
    if tif.get_named_type(None, type_name, ida_typeinf.BTF_STRUCT):
        return tif

    if tif.get_named_type(None, type_name, ida_typeinf.BTF_TYPEDEF):
        return tif

    if tif.get_named_type(None, type_name, ida_typeinf.BTF_ENUM):
        return tif

    if tif.get_named_type(None, type_name, ida_typeinf.BTF_UNION):
        return tif

    if tif := ida_typeinf.tinfo_t(type_name):
        return tif

    raise IDAError(f"Unable to retrieve {type_name} type info object")


@jsonrpc
@idaread
def get_function_by_address(
    address: Annotated[str, "Hexadecimal address (e.g., '0x401000') of the function to retrieve"],
) -> Function:
    """
    Retrieve function metadata for the function containing the given address.
    
    Returns basic function information including name, the input address (not start address), 
    and size. The address must point to a location within a function recognized by IDA.
    This is a read-only operation that does not return function code content.
    
    Note: The returned address field contains the input address, not the function's start address.
    Use decompile_function() or disassemble_function() to get actual code content.
    For arbitrary memory content, use read_memory_bytes() instead.
    """
    return get_function(parse_address(address))

@jsonrpc
@idaread
def get_current_address() -> str:
    """
    Get the memory address currently selected in the IDA interface.
    
    Returns the hexadecimal address where the user's cursor is positioned
    in the disassembly view or other IDA windows. This is a read-only operation
    that provides context about the user's current focus point.
    
    Useful for interactive analysis workflows where you need to know the
    user's current location in the binary.
    """
    return hex(idaapi.get_screen_ea())

@jsonrpc
@idaread
def get_current_function() -> Optional[Function]:
    """
    Get metadata for the function containing the user's current cursor position.
    
    Returns function information if the current cursor is positioned within
    a function recognized by IDA, otherwise returns None. This is a read-only
    operation that combines cursor position detection with function analysis.
    
    Useful for context-aware analysis where you need to work with the function
    the user is currently examining.
    """
    return get_function(idaapi.get_screen_ea(), raise_error=False)

class ConvertedNumber(TypedDict):
    decimal: str
    hexadecimal: str
    bytes: str
    ascii: Optional[str]
    binary: str

@jsonrpc
def convert_number(
    text: Annotated[str, "Number in decimal, hexadecimal (0x...), or binary (0b...) format"],
    size: Annotated[int, "Byte size constraint (1, 2, 4, 8 bytes), or 0 for auto-detection"],
) -> ConvertedNumber:
    """
    Convert a number between different numerical representations and formats.
    
    Takes a number in any supported format and returns its representation in
    decimal, hexadecimal, binary, byte array, and ASCII (if printable).
    This is a utility function that doesn't access or modify the IDA database.
    
    The size parameter specifies byte constraint (1, 2, 4, 8) or 0 for auto-detection.
    Useful for quickly converting addresses, constants, and values encountered
    during reverse engineering analysis.
    """
    try:
        value = int(text, 0)
    except ValueError:
        raise IDAError(f"Invalid number: {text}")

    # Estimate the size of the number if auto-detection requested
    if size == 0:
        size = 0
        n = abs(value)
        while n:
            size += 1
            n >>= 1
        size += 7
        size //= 8

    # Convert the number to bytes
    try:
        bytes = value.to_bytes(size, "little", signed=True)
    except OverflowError:
        raise IDAError(f"Number {text} is too big for {size} bytes")

    # Convert the bytes to ASCII
    ascii = ""
    for byte in bytes.rstrip(b"\x00"):
        if byte >= 32 and byte <= 126:
            ascii += chr(byte)
        else:
            ascii = None
            break

    return ConvertedNumber(
        decimal=str(value),
        hexadecimal=hex(value),
        bytes=bytes.hex(" "),
        ascii=ascii,
        binary=bin(value),
    )

T = TypeVar("T")

class Page(TypedDict, Generic[T]):
    data: list[T]
    next_offset: Optional[int]

def paginate(data: list[T], offset: int, count: int) -> Page[T]:
    if count == 0:
        count = len(data)
    next_offset = offset + count
    if next_offset >= len(data):
        next_offset = None
    return {
        "data": data[offset:offset + count],
        "next_offset": next_offset,
    }

def pattern_filter(data: list[T], pattern: str, key: str) -> list[T]:
    if not pattern:
        return data

    # TODO: implement /regex/ matching

    def matches(item: T) -> bool:
        return pattern.lower() in item[key].lower()
    return list(filter(matches, data))


class Global(TypedDict):
    address: str
    name: str



class Import(TypedDict):
    address: str
    imported_name: str
    module: str


class String(TypedDict):
    address: str
    length: int
    string: str




def decompile_checked(address: int) -> ida_hexrays.cfunc_t:
    if not ida_hexrays.init_hexrays_plugin():
        raise IDAError("Hex-Rays decompiler is not available")
    error = ida_hexrays.hexrays_failure_t()
    cfunc: ida_hexrays.cfunc_t = ida_hexrays.decompile_func(address, error, ida_hexrays.DECOMP_WARNINGS)
    if not cfunc:
        if error.code == ida_hexrays.MERR_LICENSE:
            raise DecompilerLicenseError("Decompiler licence is not available. Use `disassemble_function` to get the assembly code instead.")

        message = f"Decompilation failed at {hex(address)}"
        if error.str:
            message += f": {error.str}"
        if error.errea != idaapi.BADADDR:
            message += f" (address: {hex(error.errea)})"
        raise IDAError(message)
    return cfunc

@jsonrpc
@idaread
def decompile_function(
    address: Annotated[str, "Hexadecimal address (e.g., '0x401000') within the function to decompile"],
) -> str:
    """
    Generate high-level C-like pseudocode for a function using the Hex-Rays decompiler.
    
    Attempts to decompile the function containing the specified address into
    readable pseudocode. Requires a valid Hex-Rays decompiler license.
    This is a read-only operation that may take time for complex functions.
    
    Returns annotated pseudocode with line numbers and addresses for correlation
    with the original binary. If decompilation fails due to license issues,
    use disassemble_function() as an alternative.
    """
    address = parse_address(address)
    cfunc = decompile_checked(address)
    if is_window_active():
        ida_hexrays.open_pseudocode(address, ida_hexrays.OPF_REUSE)
    sv = cfunc.get_pseudocode()
    pseudocode = ""
    for i, sl in enumerate(sv):
        sl: ida_kernwin.simpleline_t
        item = ida_hexrays.ctree_item_t()
        addr = None if i > 0 else cfunc.entry_ea
        if cfunc.get_line_item(sl.line, 0, False, None, item, None):
            ds = item.dstr().split(": ")
            if len(ds) == 2:
                try:
                    addr = int(ds[0], 16)
                except ValueError:
                    pass
        line = ida_lines.tag_remove(sl.line)
        if len(pseudocode) > 0:
            pseudocode += "\n"
        if not addr:
            pseudocode += f"/* line: {i} */ {line}"
        else:
            pseudocode += f"/* line: {i}, address: {hex(addr)} */ {line}"

    return pseudocode

class DisassemblyLine(TypedDict):
    segment: NotRequired[str]
    address: str
    label: NotRequired[str]
    instruction: str
    comments: NotRequired[list[str]]

class Argument(TypedDict):
    name: str
    type: str

class DisassemblyFunction(TypedDict):
    name: str
    start_ea: str
    return_type: NotRequired[str]
    arguments: NotRequired[list[Argument]]
    stack_frame: list[dict]
    lines: list[DisassemblyLine]

@jsonrpc
@idaread
def disassemble_function(
    start_address: Annotated[str, "Hexadecimal address (e.g., '0x401000') within the function to disassemble"],
) -> DisassemblyFunction:
    """
    Generate detailed assembly disassembly for a function with comprehensive metadata.
    
    Returns complete assembly code including instructions, operands, comments,
    function arguments, return type, and stack frame information. This is a
    read-only operation that works without requiring a decompiler license.
    
    Provides more detailed analysis than basic disassembly, including symbol
    resolution, value annotations, and structural information about the function.
    Use this when decompile_function() is unavailable or when you need low-level details.
    """
    start = parse_address(start_address)
    func: ida_funcs.func_t = idaapi.get_func(start)
    if not func:
        raise IDAError(f"No function found containing address {start_address}")
    if is_window_active():
        ida_kernwin.jumpto(start)

    lines = []
    for address in ida_funcs.func_item_iterator_t(func):
        seg = idaapi.getseg(address)
        segment = idaapi.get_segm_name(seg) if seg else None

        label = idc.get_name(address, 0)
        if label and label == func.name and address == func.start_ea:
            label = None
        if label == "":
            label = None

        comments = []
        if comment := idaapi.get_cmt(address, False):
            comments += [comment]
        if comment := idaapi.get_cmt(address, True):
            comments += [comment]

        raw_instruction = idaapi.generate_disasm_line(address, 0)
        tls = ida_kernwin.tagged_line_sections_t()
        ida_kernwin.parse_tagged_line_sections(tls, raw_instruction)
        insn_section = tls.first(ida_lines.COLOR_INSN)

        operands = []
        for op_tag in range(ida_lines.COLOR_OPND1, ida_lines.COLOR_OPND8 + 1):
            op_n = tls.first(op_tag)
            if not op_n:
                break

            op: str = op_n.substr(raw_instruction)
            op_str = ida_lines.tag_remove(op)

            # Do a lot of work to add address comments for symbols
            for idx in range(len(op) - 2):
                if op[idx] != idaapi.COLOR_ON:
                    continue

                idx += 1
                if ord(op[idx]) != idaapi.COLOR_ADDR:
                    continue

                idx += 1
                addr_string = op[idx:idx + idaapi.COLOR_ADDR_SIZE]
                idx += idaapi.COLOR_ADDR_SIZE

                addr = int(addr_string, 16)

                # Find the next color and slice until there
                symbol = op[idx:op.find(idaapi.COLOR_OFF, idx)]

                if symbol == '':
                    # We couldn't figure out the symbol, so use the whole op_str
                    symbol = op_str

                comments += [f"{symbol}={addr:#x}"]

                # print its value if its type is available
                try:
                    value = get_global_variable_value_internal(addr)
                except:
                    continue

                comments += [f"*{symbol}={value}"]

            operands += [op_str]

        mnem = ida_lines.tag_remove(insn_section.substr(raw_instruction))
        instruction = f"{mnem} {', '.join(operands)}"

        line = DisassemblyLine(
            address=f"{address:#x}",
            instruction=instruction,
        )

        if len(comments) > 0:
            line.update(comments=comments)

        if segment:
            line.update(segment=segment)

        if label:
            line.update(label=label)

        lines += [line]

    prototype = func.get_prototype()
    arguments: list[Argument] = [Argument(name=arg.name, type=f"{arg.type}") for arg in prototype.iter_func()] if prototype else None

    disassembly_function = DisassemblyFunction(
        name=func.name,
        start_ea=f"{func.start_ea:#x}",
        stack_frame=get_stack_frame_variables_internal(func.start_ea),
        lines=lines
    )

    if prototype:
        disassembly_function.update(return_type=f"{prototype.get_rettype()}")

    if arguments:
        disassembly_function.update(arguments=arguments)

    return disassembly_function

class Xref(TypedDict):
    address: str
    type: str
    function: Optional[Function]






@jsonrpc
@idawrite
def set_comment(
    address: Annotated[str, "Hexadecimal address (e.g., '0x401000') where to place the comment"],
    comment: Annotated[str, "Comment text to add at the specified address"],
):
    """
    Add or update comments at a specific address in both disassembly and decompiled views.
    
    Sets comments that will be visible in IDA's disassembly window and, if possible,
    in the Hex-Rays decompiler pseudocode view. This is a write operation that
    permanently modifies the IDA database with your annotations.
    
    Comments help document your analysis findings and are essential for collaborative
    reverse engineering work. The comment will be preserved in the IDA database file.
    """
    address = parse_address(address)

    if not idaapi.set_cmt(address, comment, False):
        raise IDAError(f"Failed to set disassembly comment at {hex(address)}")

    if not ida_hexrays.init_hexrays_plugin():
        return

    # Reference: https://cyber.wtf/2019/03/22/using-ida-python-to-analyze-trickbot/
    # Check if the address corresponds to a line
    try:
        cfunc = decompile_checked(address)
    except DecompilerLicenseError:
        # We failed to decompile the function due to a decompiler license error
        return

    # Special case for function entry comments
    if address == cfunc.entry_ea:
        idc.set_func_cmt(address, comment, True)
        cfunc.refresh_func_ctext()
        return

    eamap = cfunc.get_eamap()
    if address not in eamap:
        print(f"Failed to set decompiler comment at {hex(address)}")
        return
    nearest_ea = eamap[address][0].ea

    # Remove existing orphan comments
    if cfunc.has_orphan_cmts():
        cfunc.del_orphan_cmts()
        cfunc.save_user_cmts()

    # Set the comment by trying all possible item types
    tl = idaapi.treeloc_t()
    tl.ea = nearest_ea
    for itp in range(idaapi.ITP_SEMI, idaapi.ITP_COLON):
        tl.itp = itp
        cfunc.set_user_cmt(tl, comment)
        cfunc.save_user_cmts()
        cfunc.refresh_func_ctext()
        if not cfunc.has_orphan_cmts():
            return
        cfunc.del_orphan_cmts()
        cfunc.save_user_cmts()
    print(f"Failed to set decompiler comment at {hex(address)}")

def refresh_decompiler_widget():
    widget = ida_kernwin.get_current_widget()
    if widget is not None:
        vu = ida_hexrays.get_widget_vdui(widget)
        if vu is not None:
            vu.refresh_ctext()

def refresh_decompiler_ctext(function_address: int):
    error = ida_hexrays.hexrays_failure_t()
    cfunc: ida_hexrays.cfunc_t = ida_hexrays.decompile_func(function_address, error, ida_hexrays.DECOMP_WARNINGS)
    if cfunc:
        cfunc.refresh_func_ctext()




def patch_address_assemble(
    ea: int,
    assemble: str,
) -> int:
    """Patch one instruction at address ea using Keystone (ARM64 only).

    要求：
    - 仅支持 ARM64（AArch64），不支持 ARM32；
    - 分支类 B/BL：仅支持绝对地址写法（如："b 0x184030" / "bl 0x..."），本函数自动转换为 PC 相对位移并编码；
    - 不回退到 IDA 的 Assemble。
    """
    if not KEYSTONE_AVAILABLE:
        raise IDAError("Keystone 未安装或不可用。仅支持 ARM64，且分支仅接受绝对地址写法（例如：'b 0x184030'）。")

    try:
        # 先处理 B/BL 绝对地址：b 0xXXXXXXXX / bl 0xXXXXXXXX
        import re
        m = re.match(r"^\s*(b|bl)\s+0x([0-9a-fA-F]+)\s*$", assemble)
        if m is not None:
            opcode = m.group(1).lower()
            target = int(m.group(2), 16)
            diff = target - ea
            if (diff % 4) != 0:
                raise IDAError(f"B/BL 目标未按 4 字节对齐: target={hex(target)} ea={hex(ea)}")
            imm_words = diff // 4
            if not (-(1<<25) <= imm_words <= ((1<<25)-1)):
                raise IDAError("B/BL 超出可编码范围（±2^25 字）。请改用 MOVZ/MOVK(+BR)。")
            base = 0x94000000 if opcode == 'bl' else 0x14000000
            enc = (base | (imm_words & 0x03FFFFFF)) & 0xFFFFFFFF
            ida_bytes.patch_dword(ea, enc)
            return 4

        ks = Ks(KS_ARCH_ARM64, KS_MODE_LITTLE_ENDIAN)
        # 其它指令直接用 Keystone 装配
        encoding, _ = ks.asm(assemble)
        if not encoding:
            raise IDAError(f"Keystone 未产生编码：{assemble}")
        bytes_to_patch = bytes(encoding)
    except KsError as e:
        raise IDAError(f"Keystone 装配失败（ARM64）：{assemble} | {e}")
    except Exception as e:
        raise IDAError(f"装配异常：{assemble} | {e}")

    try:
        ida_bytes.patch_bytes(ea, bytes_to_patch)
    except Exception as e:
        raise IDAError(f"写入补丁失败 @ {hex(ea)}: {e}")

    return len(bytes_to_patch)

@jsonrpc
@idawrite
def patch_address_assembles(
    address: Annotated[str, "Hexadecimal starting address (e.g., '0x401000') where to apply patches"],
    assembles: Annotated[str, "Assembly instructions separated by semicolons (e.g., 'mov eax, 1; ret')"],
) -> str:
    """
    Apply multiple assembly instruction patches at consecutive memory addresses.
    
    **WARNING: This is a destructive write operation that permanently modifies 
    the binary in memory and the IDA database.** Use with extreme caution.
    
    Assembles and patches a sequence of assembly instructions starting at the
    specified address. Each instruction overwrites the existing bytes at that
    location. Instructions are applied sequentially at consecutive addresses.
    
    This function is intended for advanced binary modification and exploit
    development. Always backup your IDA database before using this function.
    """
    ea = parse_address(address)
    assembles = [asm.strip() for asm in assembles.split(";") if asm.strip()]  # 过滤空指令
    
    if not assembles:
        raise IDAError("No valid assembly instructions provided")
    
    patched_count = 0
    for i, assemble in enumerate(assembles):
        try:
            # Ensure ea is always an integer
            if not isinstance(ea, int):
                ea = int(ea)
            patch_bytes_len = patch_address_assemble(ea, assemble)
            # Ensure patch_bytes_len is an integer
            if not isinstance(patch_bytes_len, int):
                patch_bytes_len = int(patch_bytes_len)
            ea += patch_bytes_len
            patched_count += 1
        except IDAError as e:
            raise IDAError(f"Failed to patch instruction #{i+1} '{assemble}' at address {hex(ea)}: {e.message}")
        except Exception as e:
            raise IDAError(f"Unexpected error patching instruction #{i+1} '{assemble}' at address {hex(ea)}: {str(e)}")
    
    return f"Successfully patched {patched_count} instructions"



def get_global_variable_value_internal(ea: int) -> str:
     # Get the type information for the variable
     tif = ida_typeinf.tinfo_t()
     if not ida_nalt.get_tinfo(tif, ea):
         # No type info, maybe we can figure out its size by its name
         if not ida_bytes.has_any_name(ea):
             raise IDAError(f"Failed to get type information for variable at {ea:#x}")

         size = ida_bytes.get_item_size(ea)
         if size == 0:
             raise IDAError(f"Failed to get type information for variable at {ea:#x}")
     else:
         # Determine the size of the variable
         size = tif.get_size()

     # Read the value based on the size
     if size == 0 and tif.is_array() and tif.get_array_element().is_decl_char():
         return_string = idaapi.get_strlit_contents(ea, -1, 0).decode("utf-8").strip()
         return f"\"{return_string}\""
     elif size == 1:
         return hex(ida_bytes.get_byte(ea))
     elif size == 2:
         return hex(ida_bytes.get_word(ea))
     elif size == 4:
         return hex(ida_bytes.get_dword(ea))
     elif size == 8:
         return hex(ida_bytes.get_qword(ea))
     else:
         # For other sizes, return the raw bytes
         return ' '.join(hex(x) for x in ida_bytes.get_bytes(ea, size))




class my_modifier_t(ida_hexrays.user_lvar_modifier_t):
    def __init__(self, var_name: str, new_type: ida_typeinf.tinfo_t):
        ida_hexrays.user_lvar_modifier_t.__init__(self)
        self.var_name = var_name
        self.new_type = new_type

    def modify_lvars(self, lvars):
        for lvar_saved in lvars.lvvec:
            lvar_saved: ida_hexrays.lvar_saved_info_t
            if lvar_saved.name == self.var_name:
                lvar_saved.type = self.new_type
                return True
        return False

# NOTE: This is extremely hacky, but necessary to get errors out of IDA
def parse_decls_ctypes(decls: str, hti_flags: int) -> tuple[int, str]:
    if sys.platform == "win32":
        import ctypes

        assert isinstance(decls, str), "decls must be a string"
        assert isinstance(hti_flags, int), "hti_flags must be an int"
        c_decls = decls.encode("utf-8")
        c_til = None
        ida_dll = ctypes.CDLL("ida")
        ida_dll.parse_decls.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        ida_dll.parse_decls.restype = ctypes.c_int

        messages = []

        @ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p)
        def magic_printer(fmt: bytes, arg1: bytes):
            if fmt.count(b"%") == 1 and b"%s" in fmt:
                formatted = fmt.replace(b"%s", arg1)
                messages.append(formatted.decode("utf-8"))
                return len(formatted) + 1
            else:
                messages.append(f"unsupported magic_printer fmt: {repr(fmt)}")
                return 0

        errors = ida_dll.parse_decls(c_til, c_decls, magic_printer, hti_flags)
    else:
        # NOTE: The approach above could also work on other platforms, but it's
        # not been tested and there are differences in the vararg ABIs.
        errors = ida_typeinf.parse_decls(None, decls, False, hti_flags)
        messages = []
    return errors, messages



class StackFrameVariable(TypedDict):
    name: str
    offset: str
    size: str
    type: str


def get_stack_frame_variables_internal(function_address: int) -> list[dict]:
    func = idaapi.get_func(function_address)
    if not func:
        raise IDAError(f"No function found at address {function_address}")

    members = []
    tif = ida_typeinf.tinfo_t()
    if not tif.get_type_by_tid(func.frame) or not tif.is_udt():
        return []

    udt = ida_typeinf.udt_type_data_t()
    tif.get_udt_details(udt)
    for udm in udt:
        if not udm.is_gap():
            name = udm.name
            offset = udm.offset // 8
            size = udm.size // 8
            type = str(udm.type)

            members += [StackFrameVariable(name=name,
                                           offset=hex(offset),
                                           size=hex(size),
                                           type=type)
            ]

    return members


class StructureMember(TypedDict):
    name: str
    offset: str
    size: str
    type: str

class StructureDefinition(TypedDict):
    name: str
    size: str
    members: list[StructureMember]


@jsonrpc
@idaread
def read_memory_bytes(
        memory_address: Annotated[str, "Hexadecimal address (e.g., '0x401000') to start reading from"],
        size: Annotated[int, "Number of bytes to read (must be positive integer)"]
) -> str:
    """
    Read raw bytes from memory at a specified address and return as hex string.
    
    Returns the raw byte values as space-separated hexadecimal representation.
    This is a read-only operation that can access any memory location visible
    to IDA, regardless of whether it's code, data, or unanalyzed regions.
    
    Use this for reading arbitrary memory content, binary data, or when you need
    raw bytes rather than interpreted function or variable information.
    """
    return ' '.join(f'{x:#02x}' for x in ida_bytes.get_bytes(parse_address(memory_address), size))


@jsonrpc
@idaread
def get_basic_block(
    address: Annotated[str, "Hexadecimal address (e.g., '0x401000') within the basic block to retrieve"]
) -> dict:
    """
    Get the basic block containing the specified address.
    
    Returns detailed information about the basic block including start/end addresses,
    type information, and complete disassembly of all instructions in the block.
    This is more precise than disassembling an entire function.
    
    Use this for analyzing specific code blocks, jump targets (like loc_ labels),
    or when you need focused analysis of a particular code region.
    """
    target_ea = parse_address(address)
    
    # Get the function containing this address
    func = idaapi.get_func(target_ea)
    if func is None:
        raise IDAError(f"No function found containing address {address}")
    
    # Create flow chart for the function
    flowchart = idaapi.FlowChart(func)
    
    # Find the basic block containing the target address
    target_block = None
    for block in flowchart:
        if block.start_ea <= target_ea < block.end_ea:
            target_block = block
            break
    
    if target_block is None:
        raise IDAError(f"No basic block found containing address {address}")
    
    # Get disassembly for the entire basic block
    lines = []
    current_ea = target_block.start_ea
    
    while current_ea < target_block.end_ea:
        # Get instruction at current address
        insn = idaapi.insn_t()
        if idaapi.decode_insn(insn, current_ea):
            # Get instruction mnemonic and operands
            instruction = ida_lines.generate_disasm_line(current_ea, 0)
            if instruction:
                # Clean up the instruction string
                instruction = ida_lines.tag_remove(instruction).strip()
                
                # Get any comments
                comment = ida_bytes.get_cmt(current_ea, False)
                if not comment:
                    comment = ida_bytes.get_cmt(current_ea, True)
                
                line_info = {
                    "address": f"0x{current_ea:x}",
                    "instruction": instruction,
                    "segment": ".text"
                }
                
                if comment:
                    line_info["comment"] = comment
                
                # Check if this address has a label
                label = ida_name.get_name(current_ea)
                if label and not label.startswith("sub_") and not label.startswith("loc_"):
                    line_info["label"] = label
                elif label and label.startswith("loc_"):
                    line_info["label"] = label
                
                lines.append(line_info)
            
            current_ea += insn.size
        else:
            # Couldn't decode instruction, skip to next byte
            current_ea += 1
    
    return {
        "start_ea": f"0x{target_block.start_ea:x}",
        "end_ea": f"0x{target_block.end_ea:x}", 
        "block_id": target_block.id,
        "block_type": target_block.type,
        "size": target_block.end_ea - target_block.start_ea,
        "function_name": ida_funcs.get_func_name(func.start_ea),
        "function_start": f"0x{func.start_ea:x}",
        "lines": lines
    }

@jsonrpc
@idawrite
def execute_ida_python_code(
    code: Annotated[str, "Python code to execute within IDA's Python environment with full API access"],
) -> dict:
    """
    **THE ONLY FUNCTION FOR EXECUTING ARBITRARY IDA PYTHON CODE**
    
    This is the exclusive entry point for running custom Python code within IDA's environment.
    All dynamic code execution, scripting, and custom analysis should use this function.
    
    **CRITICAL SECURITY NOTE: This function can execute ANY Python code with full IDA API access.
    It can read, modify, or corrupt the IDA database. Use with extreme caution.**
    
    Provides complete access to:
    - Full IDA Python API (ida_bytes, idc, idaapi, ida_funcs, idautils, etc.)
    - Binary data reading functions (get_byte, get_word, get_dword, get_qword, get_bytes)
    - Mathematical and bitwise operations (rol, ror, hex, bin, etc.)
    - All IDA constants and utilities
    
    Common use cases:
    - Deobfuscation algorithms: result = encrypted_value ^ key
    - Address calculations: target = base_addr + offset  
    - String decoding: decoded = decode_algorithm(encrypted_string)
    - Custom analysis scripts: analyze_function(0x401000)
    - Data extraction: values = [get_dword(addr) for addr in range(start, end, 4)]
    - Dynamic patching: ida_bytes.patch_byte(addr, new_value)
    
    Example:
        code = '''
        # Multi-step deobfuscation
        encrypted_data = get_bytes(0x402000, 16)
        key = 0xDEADBEEF
        decrypted = []
        for i, byte in enumerate(encrypted_data):
            decrypted.append(byte ^ ((key >> (i % 4) * 8) & 0xFF))
        result_string = ''.join(chr(b) for b in decrypted if 32 <= b <= 126)
        '''
    
    Returns execution results with success/error status and all user-defined variables.
    This is the ONLY way to run dynamic Python code within this MCP server.
    """
    import sys
    import io
    import traceback
    
    # 准备执行环境，包含常用的IDA APIs和工具函数
    exec_globals = {
        # 常用IDA APIs (只读操作)
        'ida_bytes': ida_bytes,
        'idc': idc,
        'idaapi': idaapi,
        'ida_funcs': ida_funcs,
        'idautils': idautils,
        
        # 便捷的数据读取函数
        'get_byte': ida_bytes.get_byte,
        'get_word': ida_bytes.get_word,
        'get_dword': ida_bytes.get_dword,
        'get_qword': ida_bytes.get_qword,
        'get_bytes': ida_bytes.get_bytes,
        
        # 常用工具函数
        'hex': hex,
        'bin': bin,
        'int': int,
        'pow': pow,
        'abs': abs,
        'min': min,
        'max': max,
        
        # 位操作工具
        'rol': lambda x, n, bits=32: ((x << n) | (x >> (bits - n))) & ((1 << bits) - 1),
        'ror': lambda x, n, bits=32: ((x >> n) | (x << (bits - n))) & ((1 << bits) - 1),
        
        # 常用常量
        'BADADDR': idaapi.BADADDR,
    }
    
    # 捕获输出
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    
    try:
        # 直接执行代码
        exec(code, exec_globals)
        
        # 预设的系统变量和模块，不包含在结果中
        system_vars = {
            'ida_bytes', 'idc', 'idaapi', 'ida_funcs', 'idautils',
            'get_byte', 'get_word', 'get_dword', 'get_qword', 'get_bytes',
            'hex', 'bin', 'int', 'pow', 'abs', 'min', 'max', 'rol', 'ror',
            'BADADDR'
        }
        
        # 返回用户定义的变量，排除系统变量和模块
        result = {}
        for k, v in exec_globals.items():
            if not k.startswith('__') and k not in system_vars:
                try:
                    # 尝试JSON序列化测试，确保值可以被序列化
                    import json
                    json.dumps(v)
                    result[k] = v
                except (TypeError, ValueError):
                    # 如果不能序列化，转换为字符串表示
                    result[k] = str(v)
        
        return {
            "success": True,
            "result": result,
            "error": None
        }
        
    except Exception as e:
        return {
            "success": False,
            "result": None,
            "error": str(e)
        }
    finally:
        sys.stdout = old_stdout


@jsonrpc
@idawrite
def get_blocks_referencing_block(
    address: Annotated[str, "Hexadecimal address within the target block"]
) -> list[dict]:
    """
    Get all basic blocks that reference (are predecessors of) the specified block.
    
    Uses IDA's native FlowChart API to find predecessor blocks.
    Returns a list of basic block information for blocks that can jump or branch
    to the target block.
    """
    try:
        target_ea = parse_address(address)
        
        # Get the function containing this address
        func = idaapi.get_func(target_ea)
        if not func:
            raise JSONRPCError(-32603, f"No function found at address {address}")
        
        # Create flow chart with predecessor support (FC_PREDS flag = 0x4)
        flowchart = idaapi.FlowChart(func, None, idaapi.FC_PREDS)
        
        # Find the target basic block
        target_block = None
        for block in flowchart:
            if block.start_ea <= target_ea < block.end_ea:
                target_block = block
                break
        
        if not target_block:
            raise JSONRPCError(-32603, f"No basic block found at address {address}")
        
        # Get all predecessor blocks
        predecessors = []
        for pred in target_block.preds():
            # Find the instruction that references the target block
            referencing_instructions = []
            ea = pred.start_ea
            while ea < pred.end_ea:
                # Check if this instruction references the target block
                refs_from = list(idautils.CodeRefsFrom(ea, 1))
                for ref in refs_from:
                    if target_block.start_ea <= ref < target_block.end_ea:
                        referencing_instructions.append({
                            "address": f"0x{ea:x}",
                            "instruction": idc.GetDisasm(ea)
                        })
                ea = idc.next_head(ea)
            
            predecessors.append({
                "block_start": f"0x{pred.start_ea:x}",
                "block_end": f"0x{pred.end_ea:x}",
                "block_id": pred.id,
                "referencing_instructions": referencing_instructions
            })
        
        return predecessors
        
    except Exception as e:
        raise JSONRPCError(-32603, f"Failed to get predecessor blocks: {str(e)}")

@jsonrpc
@idawrite
def get_blocks_referenced_by_block(
    address: Annotated[str, "Hexadecimal address within the source block"]
) -> list[dict]:
    """
    Get all basic blocks that are referenced by (are successors of) the specified block.
    
    Uses IDA's native FlowChart API to find successor blocks.
    Returns a list of basic blocks that the source block can jump or branch to.
    For indirect jumps, includes a special entry indicating unresolved target.
    """
    try:
        source_ea = parse_address(address)
        
        # Get the function containing this address
        func = idaapi.get_func(source_ea)
        if not func:
            raise JSONRPCError(-32603, f"No function found at address {address}")
        
        # Create flow chart
        flowchart = idaapi.FlowChart(func)
        
        # Find the source basic block
        source_block = None
        for block in flowchart:
            if block.start_ea <= source_ea < block.end_ea:
                source_block = block
                break
        
        if not source_block:
            raise JSONRPCError(-32603, f"No basic block found at address {address}")
        
        # Get all successor blocks
        successors = []
        for succ in source_block.succs():
            # Find the instruction in source block that references this successor
            referencing_instructions = []
            ea = source_block.start_ea
            while ea < source_block.end_ea:
                refs_from = list(idautils.CodeRefsFrom(ea, 1))
                for ref in refs_from:
                    if succ.start_ea <= ref < succ.end_ea:
                        referencing_instructions.append({
                            "address": f"0x{ea:x}",
                            "instruction": idc.GetDisasm(ea)
                        })
                ea = idc.next_head(ea)
            
            successors.append({
                "block_start": f"0x{succ.start_ea:x}",
                "block_end": f"0x{succ.end_ea:x}",
                "block_id": succ.id,
                "referencing_instructions": referencing_instructions
            })
        
        # Check for indirect jumps at the end of the block
        last_ea = idc.prev_head(source_block.end_ea)
        if last_ea != idaapi.BADADDR:
            mnem = idc.print_insn_mnem(last_ea)
            if mnem in ["BR", "BLR", "RET"]:  # ARM64 indirect branches
                successors.append({
                    "block_start": "indirect",
                    "block_end": "indirect",
                    "block_id": -1,
                    "referencing_instructions": [{
                        "address": f"0x{last_ea:x}",
                        "instruction": idc.GetDisasm(last_ea)
                    }]
                })
        
        return successors
        
    except Exception as e:
        raise JSONRPCError(-32603, f"Failed to get successor blocks: {str(e)}")


class MCP(idaapi.plugin_t):
    flags = idaapi.PLUGIN_KEEP
    comment = "MCP Plugin"
    help = "MCP"
    wanted_name = "MCP"
    wanted_hotkey = "Ctrl-Alt-M"

    def init(self):
        self.server = Server()
        hotkey = MCP.wanted_hotkey.replace("-", "+")
        if sys.platform == "darwin":
            hotkey = hotkey.replace("Alt", "Option")
        print(f"[MCP] Plugin loaded, use Edit -> Plugins -> MCP ({hotkey}) to start the server")
        return idaapi.PLUGIN_KEEP

    def run(self, args):
        self.server.start()

    def term(self):
        self.server.stop()

def PLUGIN_ENTRY():
    return MCP()
