#!/usr/bin/env python3
"""Map a Python package: files, classes, top-level functions, __all__ exports."""

import ast
import os
import sys


def walk_python_files(root: str):
    for dirpath, _, filenames in os.walk(root):
        for f in sorted(filenames):
            if f.endswith(".py"):
                yield os.path.join(dirpath, f)


def extract_definitions(filepath: str):
    try:
        with open(filepath) as fh:
            tree = ast.parse(fh.read(), filename=filepath)
    except (SyntaxError, UnicodeDecodeError):
        return [], [], None

    classes = []
    functions = []
    all_export = None

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            bases = ", ".join(
                ast.unparse(b) for b in node.bases
            )
            classes.append((node.lineno, node.name, bases))
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            functions.append((node.lineno, node.name))
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    try:
                        all_export = ast.literal_eval(node.value)
                    except (ValueError, TypeError):
                        all_export = "<dynamic>"

    return classes, functions, all_export


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "."

    files = list(walk_python_files(target))
    print(f"=== {len(files)} Python files in {target} ===")
    for f in files:
        print(f"  {f}")
    print()

    for filepath in files:
        classes, functions, all_export = extract_definitions(filepath)
        if not classes and not functions and all_export is None:
            continue

        print(f"--- {filepath} ---")
        for lineno, name, bases in classes:
            base_str = f"({bases})" if bases else ""
            print(f"  L{lineno:>4d}  class {name}{base_str}")
        for lineno, name in functions:
            print(f"  L{lineno:>4d}  def {name}()")
        if all_export is not None:
            print(f"         __all__ = {all_export}")
        print()


if __name__ == "__main__":
    main()
