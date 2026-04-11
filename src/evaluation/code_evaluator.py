"""
Code quality evaluator for Task-Oriented Success metrics.

Evaluates generated code for:
- Syntax correctness (Python)
- Static analysis (AST)
- Basic security checks (e.g., no os.system)
"""

import ast
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

@dataclass
class CodeMetrics:
    is_valid_syntax: bool
    has_imports: List[str]
    loc: int
    complexity: int  # Cyclomatic complexity approximation
    security_issues: List[str]

class CodeQualityEvaluator:
    """Evaluates code generation quality."""

    def __init__(self, language: str = "python"):
        self.language = language

    def evaluate(self, generated_text: str) -> CodeMetrics:
        """
        Evaluate code snippet.
        
        Args:
            generated_text: The LLM output containing code.
            
        Returns:
            CodeMetrics object.
        """
        code = self._extract_code(generated_text)
        if not code:
             return CodeMetrics(False, [], 0, 0, ["No code block found"])

        if self.language.lower() == "python":
            return self._evaluate_python(code)
        
        # Fallback for other languages (basic LOC count)
        return CodeMetrics(True, [], len(code.splitlines()), 0, [])

    def _extract_code(self, text: str) -> str:
        """Extract code from markdown blocks."""
        if "```" in text:
            blocks = text.split("```")
            # Return the first block that looks like code (usually index 1)
            if len(blocks) > 1:
                # Handle language tag e.g. ```python
                content = blocks[1]
                if "\n" in content:
                    return content.split("\n", 1)[1]
                return content
        return text

    def _evaluate_python(self, code: str) -> CodeMetrics:
        try:
            tree = ast.parse(code)
            
            # Imports
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for n in node.names:
                        imports.append(n.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)

            # Complexity (simple count of branching nodes)
            complexity = 1
            for node in ast.walk(tree):
                if isinstance(node, (ast.If, ast.For, ast.While, ast.ExceptHandler, ast.With)):
                    complexity += 1

            # Security check
            security_issues = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Attribute):
                        # check for subprocess.run, os.system, etc.
                        if getattr(node.func.value, 'id', '') in {'os', 'subprocess'} and node.func.attr in {'system', 'popen', 'run', 'call'}:
                            security_issues.append(f"Potentially unsafe call: {node.func.value.id}.{node.func.attr}")
            
            return CodeMetrics(
                is_valid_syntax=True,
                has_imports=imports,
                loc=len(code.splitlines()),
                complexity=complexity,
                security_issues=security_issues
            )

        except SyntaxError as e:
            return CodeMetrics(
                is_valid_syntax=False,
                has_imports=[],
                loc=len(code.splitlines()),
                complexity=0,
                security_issues=[f"SyntaxError: {str(e)}"]
            )
        except Exception as e:
            return CodeMetrics(
                is_valid_syntax=False,
                has_imports=[],
                loc=0,
                complexity=0,
                security_issues=[f"Error: {str(e)}"]
            )
