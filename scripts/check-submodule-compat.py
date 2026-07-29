#!/usr/bin/env python3
"""
Submodule Compatibility Checker for AutoQuake

This script checks for potential compatibility issues when updating git submodules.
It analyzes dependency changes, API changes, and runs tests to identify issues.

Usage:
    # 1. Fetch submodule updates (without merging)
    git submodule update --remote --no-merge

    # 2. Run this script to check compatibility
    python scripts/check-submodule-compat.py

    # 3. If all looks good, merge the updates
    git submodule update --remote --merge
"""

import subprocess
import sys
import re
import ast
from pathlib import Path
from configparser import ConfigParser
from dataclasses import dataclass, field


@dataclass
class SubmoduleInfo:
    """Information about a git submodule."""

    name: str
    path: Path
    current_commit: str = ''
    incoming_commit: str = ''
    has_updates: bool = False


@dataclass
class CompatibilityReport:
    """Report of compatibility issues for a submodule."""

    submodule: str
    dependency_changes: list[str] = field(default_factory=list)
    api_breaking_changes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    test_failures: list[str] = field(default_factory=list)


def get_all_submodules() -> list[SubmoduleInfo]:
    """Auto-detect all submodules from .gitmodules file."""
    gitmodules = Path('.gitmodules')
    if not gitmodules.exists():
        print('No .gitmodules file found.')
        return []

    config = ConfigParser()
    config.read(gitmodules)

    submodules = []
    for section in config.sections():
        if section.startswith('submodule'):
            # Extract name from section like 'submodule "autoquake/GaMMA"'
            match = re.search(r'"([^"]+)"', section)
            name = match.group(1) if match else section
            path = config.get(section, 'path')
            submodules.append(SubmoduleInfo(name=name, path=Path(path)))

    return submodules


def get_submodule_commits(submodule: SubmoduleInfo) -> SubmoduleInfo:
    """Get current and incoming commit hashes for a submodule."""
    if not submodule.path.exists():
        return submodule

    # Get current HEAD commit
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=submodule.path,
            capture_output=True,
            text=True,
            check=True,
        )
        submodule.current_commit = result.stdout.strip()[:7]
    except subprocess.CalledProcessError:
        submodule.current_commit = 'unknown'

    # Get FETCH_HEAD (incoming) commit
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'FETCH_HEAD'],
            cwd=submodule.path,
            capture_output=True,
            text=True,
            check=True,
        )
        submodule.incoming_commit = result.stdout.strip()[:7]
        submodule.has_updates = submodule.current_commit != submodule.incoming_commit
    except subprocess.CalledProcessError:
        submodule.incoming_commit = submodule.current_commit
        submodule.has_updates = False

    return submodule


def check_dependency_changes(submodule: SubmoduleInfo) -> list[str]:
    """Check for dependency changes in submodule's requirements."""
    changes = []

    req_files = [
        submodule.path / 'requirements.txt',
        submodule.path / 'pyproject.toml',
        submodule.path / 'setup.py',
    ]

    for req_file in req_files:
        if req_file.exists():
            # Get diff between current and incoming versions
            try:
                result = subprocess.run(
                    ['git', 'diff', 'HEAD', 'FETCH_HEAD', '--', req_file.name],
                    cwd=submodule.path,
                    capture_output=True,
                    text=True,
                )
                if result.stdout.strip():
                    # Parse the diff to extract meaningful changes
                    for line in result.stdout.split('\n'):
                        if line.startswith('+') and not line.startswith('+++'):
                            dep = line[1:].strip()
                            if dep and not dep.startswith('#'):
                                changes.append(f'Added/Changed: {dep}')
                        elif line.startswith('-') and not line.startswith('---'):
                            dep = line[1:].strip()
                            if dep and not dep.startswith('#'):
                                changes.append(f'Removed: {dep}')
            except subprocess.CalledProcessError:
                pass

    return changes


def find_imports_from_submodule(submodule_name: str) -> list[tuple[str, int, str]]:
    """Find all imports from a submodule in the autoquake directory."""
    imports = []
    autoquake_dir = Path('autoquake')

    if not autoquake_dir.exists():
        return imports

    # Map submodule names to import patterns
    import_patterns = {
        'autoquake/GaMMA': ['gamma', 'GaMMA'],
        'autoquake/EQNet': ['eqnet', 'EQNet', 'phasenet', 'PhaseNet'],
    }

    patterns = import_patterns.get(submodule_name, [submodule_name.split('/')[-1]])

    for py_file in autoquake_dir.rglob('*.py'):
        try:
            content = py_file.read_text(encoding='utf-8')
            for i, line in enumerate(content.split('\n'), 1):
                for pattern in patterns:
                    if f'from {pattern}' in line or f'import {pattern}' in line:
                        imports.append((str(py_file), i, line.strip()))
        except Exception:
            pass

    return imports


def check_api_changes(submodule: SubmoduleInfo) -> list[str]:
    """Check for API changes that might affect AutoQuake."""
    changes = []

    # Find what we import from this submodule
    imports = find_imports_from_submodule(submodule.name)

    if imports:
        changes.append(f'Found {len(imports)} imports from {submodule.name}:')
        for file, line, import_stmt in imports[:5]:  # Show first 5
            changes.append(f'  - {file}:{line}: {import_stmt}')
        if len(imports) > 5:
            changes.append(f'  ... and {len(imports) - 5} more')

        # Check if there are changes in Python files
        try:
            result = subprocess.run(
                ['git', 'diff', '--name-only', 'HEAD', 'FETCH_HEAD'],
                cwd=submodule.path,
                capture_output=True,
                text=True,
            )
            changed_files = [f for f in result.stdout.strip().split('\n') if f.endswith('.py')]
            if changed_files:
                changes.append(f'\nPython files changed in {submodule.name}:')
                for f in changed_files[:10]:
                    changes.append(f'  - {f}')
                if len(changed_files) > 10:
                    changes.append(f'  ... and {len(changed_files) - 10} more')
        except subprocess.CalledProcessError:
            pass

    return changes


def run_tests() -> tuple[bool, list[str]]:
    """Run the test suite and return results."""
    failures = []

    print('\nRunning tests with current submodule versions...')
    result = subprocess.run(
        ['python', '-m', 'pytest', 'tests/', '-v', '--tb=short', '-x'],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        # Extract failure information
        for line in result.stdout.split('\n'):
            if 'FAILED' in line or 'ERROR' in line:
                failures.append(line.strip())

        # Also include stderr if there were import errors
        if 'ModuleNotFoundError' in result.stderr or 'ImportError' in result.stderr:
            for line in result.stderr.split('\n'):
                if 'Error' in line:
                    failures.append(line.strip())

    return result.returncode == 0, failures


def print_report(reports: list[CompatibilityReport], test_passed: bool, test_failures: list[str]):
    """Print the compatibility report."""
    print('\n' + '=' * 60)
    print('SUBMODULE COMPATIBILITY REPORT')
    print('=' * 60)

    # Summarize submodules with updates
    submodules_with_updates = [r for r in reports if r.dependency_changes or r.api_breaking_changes]

    if not submodules_with_updates and test_passed:
        print('\n All submodules are compatible. Safe to merge.')
        return

    for report in reports:
        if report.dependency_changes or report.api_breaking_changes or report.warnings:
            print(f'\n[{report.submodule}]')
            print('-' * 45)

            if report.dependency_changes:
                print('\n DEPENDENCY CHANGES:')
                for change in report.dependency_changes:
                    print(f'   {change}')

            if report.api_breaking_changes:
                print('\n API CHANGES (review required):')
                for change in report.api_breaking_changes:
                    print(f'   {change}')

            if report.warnings:
                print('\n Warnings:')
                for warning in report.warnings:
                    print(f'   {warning}')

    # Test results
    print('\n' + '=' * 60)
    print('TEST RESULTS')
    print('=' * 60)

    if test_passed:
        print('\n All tests passed!')
    else:
        print('\n Some tests failed:')
        for failure in test_failures:
            print(f'   {failure}')

    # Summary and recommendations
    print('\n' + '=' * 60)
    print('RECOMMENDATIONS')
    print('=' * 60)

    if test_passed and not submodules_with_updates:
        print('\n Safe to merge submodule updates.')
        print('   git submodule update --remote --merge')
    else:
        print('\n Before merging:')
        if not test_passed:
            print('   1. Fix failing tests')
        if submodules_with_updates:
            print('   2. Review API changes and update code if needed')
        print('\n After fixes:')
        print('   git submodule update --remote --merge')


def main():
    """Main entry point."""
    print('=' * 60)
    print('AutoQuake Submodule Compatibility Checker')
    print('=' * 60)

    # Get all submodules
    submodules = get_all_submodules()

    if not submodules:
        print('No submodules found.')
        return

    print(f'\nDetected submodules: {", ".join(s.name for s in submodules)}')

    # Check each submodule
    reports = []
    for sm in submodules:
        sm = get_submodule_commits(sm)

        if sm.has_updates:
            print(f'\n[{sm.name}] {sm.current_commit} -> {sm.incoming_commit}')
        else:
            print(f'\n[{sm.name}] {sm.current_commit} (no updates)')
            continue

        report = CompatibilityReport(submodule=sm.name)

        # Check dependencies
        report.dependency_changes = check_dependency_changes(sm)

        # Check API changes
        report.api_breaking_changes = check_api_changes(sm)

        reports.append(report)

    # Run tests
    test_passed, test_failures = run_tests()

    # Print final report
    print_report(reports, test_passed, test_failures)


if __name__ == '__main__':
    main()
