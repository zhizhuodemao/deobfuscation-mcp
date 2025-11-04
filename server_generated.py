# NOTE: This file has been automatically generated, do not modify!
# Architecture based on https://github.com/mrexodia/ida-pro-mcp (MIT License)
import sys
if sys.version_info >= (3, 12):
    from typing import Annotated, Optional, TypedDict, Generic, TypeVar, NotRequired
else:
    from typing_extensions import Annotated, Optional, TypedDict, Generic, TypeVar, NotRequired
from pydantic import Field

T = TypeVar("T")

class Metadata(TypedDict):
    path: str
    module: str
    base: str
    size: str
    md5: str
    sha256: str
    crc32: str
    filesize: str

class Function(TypedDict):
    address: str
    name: str
    size: str

class ConvertedNumber(TypedDict):
    decimal: str
    hexadecimal: str
    bytes: str
    ascii: Optional[str]
    binary: str

class Page(TypedDict, Generic[T]):
    data: list[T]
    next_offset: Optional[int]

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

class Xref(TypedDict):
    address: str
    type: str
    function: Optional[Function]

class StackFrameVariable(TypedDict):
    name: str
    offset: str
    size: str
    type: str

class StructureMember(TypedDict):
    name: str
    offset: str
    size: str
    type: str

class StructureDefinition(TypedDict):
    name: str
    size: str
    members: list[StructureMember]

@mcp.tool()
def get_metadata() -> Metadata:
    """
    Retrieve comprehensive metadata about the currently loaded IDA database.
    
    Returns detailed information about the binary file including file paths, 
    base addresses, size, and cryptographic hashes. This is a read-only operation
    that provides essential context for reverse engineering analysis.
    
    Use this as a starting point to understand the loaded binary's characteristics.
    """
    return make_jsonrpc_request('get_metadata')

@mcp.tool()
def get_function_by_address(address: Annotated[str, Field(description="Hexadecimal address (e.g., '0x401000') of the function to retrieve")]) -> Function:
    """
    Retrieve function metadata for the function containing the given address.
    
    Returns basic function information including name, the input address (not start address), 
    and size. The address must point to a location within a function recognized by IDA.
    This is a read-only operation that does not return function code content.
    
    Note: The returned address field contains the input address, not the function's start address.
    Use decompile_function() or disassemble_function() to get actual code content.
    For arbitrary memory content, use read_memory_bytes() instead.
    """
    return make_jsonrpc_request('get_function_by_address', address)

@mcp.tool()
def get_current_address() -> str:
    """
    Get the memory address currently selected in the IDA interface.
    
    Returns the hexadecimal address where the user's cursor is positioned
    in the disassembly view or other IDA windows. This is a read-only operation
    that provides context about the user's current focus point.
    
    Useful for interactive analysis workflows where you need to know the
    user's current location in the binary.
    """
    return make_jsonrpc_request('get_current_address')

@mcp.tool()
def get_current_function() -> Optional[Function]:
    """
    Get metadata for the function containing the user's current cursor position.
    
    Returns function information if the current cursor is positioned within
    a function recognized by IDA, otherwise returns None. This is a read-only
    operation that combines cursor position detection with function analysis.
    
    Useful for context-aware analysis where you need to work with the function
    the user is currently examining.
    """
    return make_jsonrpc_request('get_current_function')

@mcp.tool()
def convert_number(text: Annotated[str, Field(description='Number in decimal, hexadecimal (0x...), or binary (0b...) format')], size: Annotated[int, Field(description='Byte size constraint (1, 2, 4, 8 bytes), or 0 for auto-detection')]) -> ConvertedNumber:
    """
    Convert a number between different numerical representations and formats.
    
    Takes a number in any supported format and returns its representation in
    decimal, hexadecimal, binary, byte array, and ASCII (if printable).
    This is a utility function that doesn't access or modify the IDA database.
    
    The size parameter specifies byte constraint (1, 2, 4, 8) or 0 for auto-detection.
    Useful for quickly converting addresses, constants, and values encountered
    during reverse engineering analysis.
    """
    return make_jsonrpc_request('convert_number', text, size)

@mcp.tool()
def decompile_function(address: Annotated[str, Field(description="Hexadecimal address (e.g., '0x401000') within the function to decompile")]) -> str:
    """
    Generate high-level C-like pseudocode for a function using the Hex-Rays decompiler.
    
    Attempts to decompile the function containing the specified address into
    readable pseudocode. Requires a valid Hex-Rays decompiler license.
    This is a read-only operation that may take time for complex functions.
    
    Returns annotated pseudocode with line numbers and addresses for correlation
    with the original binary. If decompilation fails due to license issues,
    use disassemble_function() as an alternative.
    """
    return make_jsonrpc_request('decompile_function', address)

@mcp.tool()
def disassemble_function(start_address: Annotated[str, Field(description="Hexadecimal address (e.g., '0x401000') within the function to disassemble")]) -> DisassemblyFunction:
    """
    Generate detailed assembly disassembly for a function with comprehensive metadata.
    
    Returns complete assembly code including instructions, operands, comments,
    function arguments, return type, and stack frame information. This is a
    read-only operation that works without requiring a decompiler license.
    
    Provides more detailed analysis than basic disassembly, including symbol
    resolution, value annotations, and structural information about the function.
    Use this when decompile_function() is unavailable or when you need low-level details.
    """
    return make_jsonrpc_request('disassemble_function', start_address)

@mcp.tool()
def set_comment(address: Annotated[str, Field(description="Hexadecimal address (e.g., '0x401000') where to place the comment")], comment: Annotated[str, Field(description='Comment text to add at the specified address')]):
    """
    Add or update comments at a specific address in both disassembly and decompiled views.
    
    Sets comments that will be visible in IDA's disassembly window and, if possible,
    in the Hex-Rays decompiler pseudocode view. This is a write operation that
    permanently modifies the IDA database with your annotations.
    
    Comments help document your analysis findings and are essential for collaborative
    reverse engineering work. The comment will be preserved in the IDA database file.
    """
    return make_jsonrpc_request('set_comment', address, comment)

@mcp.tool()
def patch_address_assembles(address: Annotated[str, Field(description="Hexadecimal starting address (e.g., '0x401000') where to apply patches")], assembles: Annotated[str, Field(description="Assembly instructions separated by semicolons (e.g., 'mov eax, 1; ret')")]) -> str:
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
    return make_jsonrpc_request('patch_address_assembles', address, assembles)

@mcp.tool()
def read_memory_bytes(memory_address: Annotated[str, Field(description="Hexadecimal address (e.g., '0x401000') to start reading from")], size: Annotated[int, Field(description='Number of bytes to read (must be positive integer)')]) -> str:
    """
    Read raw bytes from memory at a specified address and return as hex string.
    
    Returns the raw byte values as space-separated hexadecimal representation.
    This is a read-only operation that can access any memory location visible
    to IDA, regardless of whether it's code, data, or unanalyzed regions.
    
    Use this for reading arbitrary memory content, binary data, or when you need
    raw bytes rather than interpreted function or variable information.
    """
    return make_jsonrpc_request('read_memory_bytes', memory_address, size)

@mcp.tool()
def get_basic_block(address: Annotated[str, Field(description="Hexadecimal address (e.g., '0x401000') within the basic block to retrieve")]) -> dict:
    """
    Get the basic block containing the specified address.
    
    Returns detailed information about the basic block including start/end addresses,
    type information, and complete disassembly of all instructions in the block.
    This is more precise than disassembling an entire function.
    
    Use this for analyzing specific code blocks, jump targets (like loc_ labels),
    or when you need focused analysis of a particular code region.
    """
    return make_jsonrpc_request('get_basic_block', address)

@mcp.tool()
def execute_ida_python_code(code: Annotated[str, Field(description="Python code to execute within IDA's Python environment with full API access")]) -> dict:
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
    return make_jsonrpc_request('execute_ida_python_code', code)

@mcp.tool()
def get_blocks_referencing_block(address: Annotated[str, Field(description='Hexadecimal address within the target block')]) -> list[dict]:
    """
    Get all basic blocks that reference (are predecessors of) the specified block.
    
    Uses IDA's native FlowChart API to find predecessor blocks.
    Returns a list of basic block information for blocks that can jump or branch
    to the target block.
    """
    return make_jsonrpc_request('get_blocks_referencing_block', address)

@mcp.tool()
def get_blocks_referenced_by_block(address: Annotated[str, Field(description='Hexadecimal address within the source block')]) -> list[dict]:
    """
    Get all basic blocks that are referenced by (are successors of) the specified block.
    
    Uses IDA's native FlowChart API to find successor blocks.
    Returns a list of basic blocks that the source block can jump or branch to.
    For indirect jumps, includes a special entry indicating unresolved target.
    """
    return make_jsonrpc_request('get_blocks_referenced_by_block', address)

