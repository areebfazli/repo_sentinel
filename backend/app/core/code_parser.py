from pathlib import Path
from typing import List, Dict, Any, Optional
import tree_sitter_python
import tree_sitter_javascript
import tree_sitter_go
import tree_sitter_java
from tree_sitter import Language, Parser

class CodeParser:
    def __init__(self):
        """Initialize the AST parsers for supported languages."""
        # Load languages
        self.languages = {
            ".py": Language(tree_sitter_python.language()),
            ".js": Language(tree_sitter_javascript.language()),
            ".ts": Language(tree_sitter_javascript.language()), # Fallback TS to JS parser for now
            ".go": Language(tree_sitter_go.language()),
            ".java": Language(tree_sitter_java.language())
        }
        
        # Queries to extract functions/methods for each language
        # We extract the entire function block to embed
        self.queries = {
            ".py": "(function_definition) @function",
            ".js": """
                (function_declaration) @function
                (method_definition) @function
                (arrow_function) @function
            """,
            ".ts": """
                (function_declaration) @function
                (method_definition) @function
                (arrow_function) @function
            """,
            ".go": """
                (function_declaration) @function
                (method_declaration) @function
            """,
            ".java": "(method_declaration) @function"
        }

    def parse_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Parse a single file and extract all functions.
        Returns a list of dictionaries containing the function code and metadata.
        """
        ext = file_path.suffix.lower()
        if ext not in self.languages:
            return [] # Unsupported language
            
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source_code = f.read()
        except Exception:
            # Skip unreadable or binary files
            return []

        parser = Parser()
        lang = self.languages[ext]
        parser.set_language(lang)
        
        tree = parser.parse(bytes(source_code, "utf8"))
        query = lang.query(self.queries[ext])
        captures = query.captures(tree.root_node)
        
        functions = []
        for node, capture_name in captures:
            # Extract the raw text of the function
            func_bytes = source_code.encode("utf8")[node.start_byte:node.end_byte]
            func_text = func_bytes.decode("utf8")
            
            functions.append({
                "file_path": str(file_path),
                "language": ext[1:], # remove the dot
                "start_line": node.start_point[0] + 1,
                "end_line": node.end_point[0] + 1,
                "code": func_text
            })
            
        return functions

    def parse_directory(self, dir_path: Path) -> List[Dict[str, Any]]:
        """
        Recursively walk a directory and extract functions from all supported files.
        """
        all_functions = []
        
        # Avoid traversing common ignored directories
        ignore_dirs = {".git", "node_modules", "venv", "env", "__pycache__", "build", "dist"}
        
        for root, dirs, files in os.walk(dir_path):
            # Mutate dirs in-place to skip ignored directories
            dirs[:] = [d for d in dirs if d not in ignore_dirs]
            
            for file in files:
                file_path = Path(root) / file
                if file_path.suffix.lower() in self.languages:
                    funcs = self.parse_file(file_path)
                    # Convert absolute paths to relative paths for better indexing
                    for f in funcs:
                        f["file_path"] = str(file_path.relative_to(dir_path))
                    all_functions.extend(funcs)
                    
        return all_functions

# Need os for os.walk
import os
