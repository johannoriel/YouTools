import os
import json
import importlib
import streamlit as st
from typing import List, Dict, Any, Set
from dotenv import load_dotenv
from lib.global_vars import translations, t
import torch
# https://discuss.streamlit.io/t/error-in-torch-with-streamlit/90908/5
torch.classes.__path__ = []
import pandas as pd
import pytest
import inspect

# Constants
CONFIG_FILE = "config.json"
CORE_PLUGINS = {'common', 'llm'}  # Add your essential plugins here


# Unit test decorator
def unit_test(func):
    """Decorator to mark methods as unit tests."""
    print(f"Marking {func.__name__} as unit test")
    func.is_unit_test = True
    return func


def list_directories(directory):
    """Liste les dossiers dans le répertoire donné."""
    directories = []
    try:
        for item in os.listdir(directory):
            full_path = os.path.join(directory, item)
            if os.path.isdir(full_path):
                directories.append((item, full_path))
        directories.sort(key=lambda x: x[0])  # Trier par nom
        return directories
    except OSError:
        return []


def load_config() -> Dict[str, Any]:
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_config(config: Dict[str, Any]):
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)


def set_lang(language):
    st.session_state.lang = language


def t(key: str) -> str:
    return translations[st.session_state.lang].get(key, key)


class Widget:
    def __init__(self, name, prefix, plugin_manager):
        self.prefix = prefix
        self.name = name
        self.plugin_manager = plugin_manager

    def process_with_llm(self, prompt: str, sysprompt: str = None, context: str = None, repeat_on_failure: bool = True, number_repeat: int = 2) -> str:
        llm = self.plugin_manager.get_plugin('llm')
        if sysprompt is None:
            sysprompt = self.plugin_manager.config['llm']['llm_sys_prompt']
        if context is None:
            context = ""
        response = llm.process_with_llm(
            prompt, sysprompt, context, repeat_on_failure, number_repeat)
        return response

    def work_dir(self):
        return os.path.expanduser(self.plugin_manager.config['common']['work_directory'])


class Plugin:
    def __init__(self, name, plugin_manager):
        self.name = name
        self.plugin_manager = plugin_manager

    def get_config_fields(self) -> Dict[str, Any]:
        return {}

    def get_config(self, config_value):
        return self.plugin_manager.config[self.name][config_value]

    def get_config_ui(self, config):
        updated_config = {}
        for field, params in self.get_config_fields().items():
            print(params['label'])
            if params['type'] == 'select':
                updated_config[field] = st.selectbox(
                    params['label'],
                    options=[option[0] for option in params['options']],
                    format_func=lambda x: dict(params['options'])[x],
                    index=[option[0] for option in params['options']].index(
                        config.get(field, params['default']))
                )
            elif params['type'] == 'textarea':
                updated_config[field] = st.text_area(
                    params['label'],
                    value=config.get(field, params['default'])
                )
            else:
                updated_config[field] = st.text_input(
                    params['label'],
                    value=config.get(field, params['default']),
                    type="password" if field.startswith("pass") else "default"
                )
        return updated_config

    def get_tabs(self) -> List[Dict[str, Any]]:
        return []

    def run(self, config: Dict[str, Any]):
        pass

    def get_sidebar_config_ui(self, expander, config: Dict[str, Any]) -> Dict[str, Any]:
        return {}

    def has_tests(self) -> bool:
        return False

    # Helper
    def process_with_llm(self, prompt: str, sysprompt: str = None, context: str = None, repeat_on_failure: bool = True, number_repeat: int = 2) -> str:
        llm = self.plugin_manager.get_plugin('llm')
        if sysprompt is None:
            sysprompt = self.plugin_manager.config['llm']['llm_sys_prompt']
        if context is None:
            context = ""
        response = llm.process_with_llm(
            prompt, sysprompt, context, repeat_on_failure, number_repeat)
        return response

    # Helper (modified to support test mode)
    def work_dir(self):
        if st.session_state.get('test_mode', False):
            return os.path.expanduser(self.plugin_manager.config['common']['test_directory'])
        return os.path.expanduser(self.plugin_manager.config['common']['work_directory'])


class PluginManager:
    def __init__(self, config):
        self.plugins: Dict[str, Plugin] = {}
        self.starred_plugins: Set[str] = set()
        self.config = config
        self.available_plugins = self._scan_plugins_directory()

    def _scan_plugins_directory(self) -> Set[str]:
        """Scan the plugins directory and return available plugin names without loading them."""
        plugins_dir = 'plugins'
        return {filename[:-3] for filename in os.listdir(plugins_dir)
                if filename.endswith('.py') and not filename.startswith('__')}

    def load_plugin(self, plugin_name: str) -> None:
        """Load a single plugin by name."""
        if plugin_name not in self.plugins and plugin_name in self.available_plugins:
            module = importlib.import_module(f'plugins.{plugin_name}')
            plugin_class = getattr(module, f'{plugin_name.capitalize()}Plugin')
            self.plugins[plugin_name] = plugin_class(plugin_name, self)

    def load_core_plugins(self):
        """Load only core plugins that are required at startup."""
        for plugin_name in CORE_PLUGINS:
            self.load_plugin(plugin_name)

    def get_plugin(self, plugin_name: str) -> Plugin:
        """Get a plugin, loading it if necessary."""
        if plugin_name not in self.plugins:
            self.load_plugin(plugin_name)
        return self.plugins.get(plugin_name)

    def get_all_config_ui(self, config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Load all plugins when accessing configurations."""
        for plugin_name in self.available_plugins:
            if plugin_name not in self.plugins:
                self.load_plugin(plugin_name)

        all_ui = {}
        for plugin_name, plugin in sorted(self.plugins.items()):
            with st.expander(f"{'⭐ ' if plugin_name in self.starred_plugins else ''}{t('configurations')} {plugin_name}"):
                all_ui[plugin_name] = plugin.get_config_ui(
                    config.get(plugin_name, {}))
                if st.button(f"{'Unstar' if plugin_name in self.starred_plugins else 'Star'} {plugin_name}"):
                    if plugin_name in self.starred_plugins:
                        self.starred_plugins.remove(plugin_name)
                    else:
                        self.starred_plugins.add(plugin_name)
                    self.save_starred_plugins(config)
                    st.rerun()
        return all_ui

    def get_sidebar_config_for_core_plugins(self, expander, config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Get sidebar configuration for core plugins only."""
        sidebar_configs = {}
        for plugin_name in CORE_PLUGINS:
            plugin = self.get_plugin(plugin_name)
            if plugin:
                sidebar_config = plugin.get_sidebar_config_ui(expander, config)
                if sidebar_config:
                    sidebar_configs[plugin_name] = sidebar_config
        return sidebar_configs

    def get_all_tabs(self) -> List[Dict[str, Any]]:
        """Get all available tabs without loading the plugins."""
        all_tabs = []
        for plugin_name in self.available_plugins:
            # Create a placeholder tab info that will be replaced when the plugin is loaded
            tab = {
                'plugin': plugin_name,
                'name': plugin_name.capitalize(),
                'id': plugin_name,
                'starred': plugin_name in self.starred_plugins
            }
            all_tabs.append(tab)

        # For loaded plugins, get the actual tab information
        for plugin_name, plugin in self.plugins.items():
            tabs = plugin.get_tabs()
            for tab in tabs:
                tab['id'] = plugin_name if 'id' not in tab else tab['id']
                tab['starred'] = plugin_name in self.starred_plugins
            # Replace the placeholder with actual tabs
            all_tabs = [t for t in all_tabs if t['plugin']
                        != plugin_name] + tabs

        return all_tabs

    def load_starred_plugins(self, config: Dict[str, Any]):
        self.starred_plugins = set(config.get('starred_plugins', []))

    def save_starred_plugins(self, config: Dict[str, Any]):
        config['starred_plugins'] = list(self.starred_plugins)
        save_config(config)

    def save_config(self, config: Dict[str, Any]):
        save_config(config)

    def run_plugin(self, plugin_name: str, config: Dict[str, Any]):
        """Run a plugin, loading it if necessary."""
        plugin = self.get_plugin(plugin_name)
        if plugin:
            plugin.run(config)


def parse_pytest_output(reports: List[pytest.TestReport]) -> List[Dict[str, str]]:
    tests = []
    print(f"Parsing {len(reports)} test reports")
    for report in reports:
        if report.when == 'call' and report.outcome in ('passed', 'failed', 'skipped'):
            test_name = report.nodeid
            status = report.outcome.upper()
            tests.append({'test': test_name, 'status': status})
            print(f"Test: {test_name}, Status: {status}")
    return tests


def run_all_tests(plugin_manager):
    print("Starting run_all_tests")
    results = {}
    test_dir = 'tests'
    os.makedirs(test_dir, exist_ok=True)
    print(f"Ensured test directory exists: {test_dir}")

    # List contents of tests/ directory for debugging
    test_files = os.listdir(test_dir)
    print(f"Files in {test_dir}: {test_files}")
    feature_files = [f for f in test_files if f.endswith('.feature')]
    print(f"Feature files in {test_dir}: {feature_files}")

    project_root = os.path.abspath(os.path.dirname(__file__))
    os.environ['PYTHONPATH'] = f"{project_root}:{os.environ.get('PYTHONPATH', '')}"
    print(f"Set PYTHONPATH to: {os.environ['PYTHONPATH']}")

    # Custom pytest plugin to collect test reports
    class TestReportCollector:
        def __init__(self):
            self.reports = []

        def pytest_runtest_logreport(self, report):
            self.reports.append(report)
            print(f"Collected report for {report.nodeid}: {report.outcome}")

    collector = TestReportCollector()

    # Run unit tests for all plugins
    for plugin_name in plugin_manager.available_plugins:
        print(f"Processing plugin: {plugin_name}")
        plugin = plugin_manager.get_plugin(plugin_name)
        if plugin.has_tests():
            print(f"Plugin {plugin_name} has tests")
            test_methods = []
            for name, method in inspect.getmembers(plugin, predicate=inspect.ismethod):
                if getattr(method, 'is_unit_test', False):
                    test_methods.append(name)
                    print(f"Found unit test method: {name}")

            if test_methods:
                print(f"Found {len(test_methods)} unit tests for {plugin_name}")
                # Create a temporary test file
                test_file = os.path.join(test_dir, f'test_{plugin_name}_unit.py')
                module_code = f"""
from plugins.{plugin_name} import {plugin_name.capitalize()}Plugin
from app import PluginManager

plugin_manager = PluginManager({{}})
plugin = {plugin_name.capitalize()}Plugin('{plugin_name}', plugin_manager)
"""
                for test_name in test_methods:
                    module_code += f"""
def test_{test_name}():
    plugin.{test_name}()
"""
                print(f"Writing test module to {test_file}:\n{module_code}")
                try:
                    with open(test_file, 'w') as f:
                        f.write(module_code)
                    pytest_args = ['-v', '--tb=short', test_file]
                    print(f"Running pytest with args: {pytest_args}")
                    result = pytest.main(pytest_args, plugins=[collector])
                    print(f"Pytest result for {plugin_name} unit tests: {result}")
                    results[plugin_name] = parse_pytest_output(collector.reports)
                    print(f"Unit test results for {plugin_name}: {results[plugin_name]}")
                    collector.reports = []  # Reset for next plugin
                except Exception as e:
                    print(f"Error running unit tests for {plugin_name}: {str(e)}")
                    results[plugin_name] = [{'test': f'Error running unit tests for {plugin_name}', 'status': str(e)}]
                finally:
                    if os.path.exists(test_file):
                        os.remove(test_file)
                        print(f"Cleaned up {test_file}")
            else:
                print(f"No unit tests found for {plugin_name}")
                results[plugin_name] = []

    # Run BDD tests in tests/ directory
    print(f"Running BDD tests in {test_dir}")
    try:
        pytest_args = ['-v', '--tb=short',  test_dir]
        print(f"Running pytest with args: {pytest_args}")
        result = pytest.main(pytest_args, plugins=[collector])
        print(f"Pytest result for BDD tests: {result}")
        bdd_results = parse_pytest_output(collector.reports)
        print(f"BDD test results: {bdd_results}")
        # Check for BDD report file
        if os.path.exists('tests/bdd_report.json'):
            with open('tests/bdd_report.json', 'r') as f:
                print(f"BDD report content: {f.read()}")
        else:
            print("No BDD report generated")
        # Distribute BDD results to plugins based on naming
        for plugin_name in plugin_manager.available_plugins:
            if plugin_name not in results:
                results[plugin_name] = []
            plugin_bdd_results = [r for r in bdd_results if f'test_{plugin_name}' in r['test'] or plugin_name in r['test']]
            results[plugin_name].extend(plugin_bdd_results)
            print(f"BDD results for {plugin_name}: {plugin_bdd_results}")
    except Exception as e:
        print(f"Error running BDD tests: {str(e)}")
        for plugin_name in plugin_manager.available_plugins:
            if plugin_name not in results:
                results[plugin_name] = []
            results[plugin_name].append({'test': 'Error running BDD tests', 'status': str(e)})

    st.session_state['test_results'] = results
    print(f"Final test results: {results}")
    st.success("Tests completed.")


def main():
    try:
        # Load configuration
        load_dotenv()
        config = load_config()
        # Initialize test_directory if not present
        if 'test_directory' not in config.get('common', {}):
            config['common']['test_directory'] = os.path.expanduser('~/test_work_dir')
            save_config(config)
        # Initialize language
        if 'lang' not in st.session_state:
            st.session_state.lang = config['common']['language']

        if 'presentation_mode' not in st.session_state or ('presentation_mode' in st.session_state and not st.session_state.presentation_mode):
            st.set_page_config(page_title="YoutTools",
                            layout="wide", initial_sidebar_state="expanded")
            st.title(t("page_title"))
        else:
            st.set_page_config(page_title="YoutTools", layout="wide",
                            initial_sidebar_state="collapsed")

        # Initialize plugin manager and load core plugins only
        plugin_manager = PluginManager(config)
        plugin_manager.load_core_plugins()
        plugin_manager.load_starred_plugins(config)

        # Create tabs
        tabs = [{"id": "configurations", "name": t(
            "configurations")}, {"id": "tests", "name": t("tests")}] + plugin_manager.get_all_tabs()

        expander = st.sidebar.expander("Configuration", expanded=True)

        # Language selection
        new_lang = expander.selectbox(
            "Choose your language / Choisissez votre langue",
            options=["en", "fr"],
            index=["en", "fr"].index(st.session_state.lang),
            key="lang_selector"
        )

        if new_lang != st.session_state.lang:
            st.session_state.lang = new_lang
            st.rerun()

        # Work directory selector
        expander.subheader(t("work_directory"))
        current_work_dir = os.path.expanduser(config['common']['work_directory'])
        directories = list_directories(current_work_dir)
        parent_dir = os.path.dirname(current_work_dir)
        dir_options = [current_work_dir, parent_dir] + [d[1] for d in directories]
        dir_labels = [current_work_dir, "Parent: " + os.path.basename(parent_dir)] + [os.path.basename(d[0]) for d in directories]

        selected_dir = expander.selectbox(
            t("work_directory"),
            options=dir_options,
            format_func=lambda x: dir_labels[dir_options.index(x)],
            key="work_dir_selector"
        )

        if selected_dir != current_work_dir:
            config['common']['work_directory'] = selected_dir
            save_config(config)
            st.rerun()

        # Test directory selector
        expander.subheader(t("test_directory"))
        current_test_dir = os.path.expanduser(config['common']['test_directory'])
        test_directories = list_directories(current_test_dir)
        test_parent_dir = os.path.dirname(current_test_dir)
        test_dir_options = [current_test_dir, test_parent_dir] + [d[1] for d in test_directories]
        test_dir_labels = [current_test_dir, "Parent: " + os.path.basename(test_parent_dir)] + [os.path.basename(d[0]) for d in test_directories]

        selected_test_dir = expander.selectbox(
            t("test_directory"),
            options=test_dir_options,
            format_func=lambda x: test_dir_labels[test_dir_options.index(x)],
            key="test_dir_selector"
        )

        if selected_test_dir != current_test_dir:
            config['common']['test_directory'] = selected_test_dir
            save_config(config)
            st.rerun()

        # Test Mode checkbox
        test_mode = expander.checkbox(t("test_mode"), value=st.session_state.get('test_mode', False))
        st.session_state['test_mode'] = test_mode

        # Handle core plugins sidebar configuration
        core_sidebar_configs = plugin_manager.get_sidebar_config_for_core_plugins(
            expander, config)
        for plugin_name, sidebar_config in core_sidebar_configs.items():
            for key, value in sidebar_config.items():
                config.setdefault(plugin_name, {})[key] = value

        # Ajouter le bouton "Clean session" dans la barre latérale
        col1, col2 = expander.columns([1, 1])
        if col1.button(t("Clean session")):
            tab = st.session_state.selected_tab_id
            st.session_state.clear()  # Cela réinitialise st.session_state
            st.session_state.selected_tab_id = tab
            st.rerun()  # Relancer l'application pour refléter les changements
        if col2.button(t("Rerun")):
            st.rerun()

        # Add "Run Tests" button
        if expander.button(t("run_tests")):
            run_all_tests(plugin_manager)

        # Initialize selected tab
        if 'selected_tab_id' not in st.session_state:
            st.session_state.selected_tab_id = "directpublish"

        # Sort and display tabs
        sorted_tabs = sorted(tabs, key=lambda x: (
            not x.get('starred', False), x['name']))
        tab_names = [
            f"{'⭐ ' if tab.get('starred', False) else ''}{tab['name']}" for tab in sorted_tabs]

        selected_tab_index = [tab["id"] for tab in sorted_tabs].index(
            st.session_state.selected_tab_id)
        selected_tab = expander.radio(
            t("navigation"), tab_names, index=selected_tab_index, key="tab_selector")

        new_selected_tab_id = next(
            tab["id"] for tab in sorted_tabs if f"{'⭐ ' if tab.get('starred', False) else ''}{tab['name']}" == selected_tab)

        if new_selected_tab_id != st.session_state.selected_tab_id:
            st.session_state.selected_tab_id = new_selected_tab_id
            st.rerun()

        # Handle selected tab
        if st.session_state.selected_tab_id == "configurations":
            st.header(t("configurations"))
            all_config_ui = plugin_manager.get_all_config_ui(config)

            for plugin_name, ui_config in all_config_ui.items():
                config[plugin_name] = ui_config

            if st.button(t("save_button")):
                save_config(config)
                st.success(t("success_message"))
        elif st.session_state.selected_tab_id == "tests":
            st.header(t("test_results"))
            if 'test_results' in st.session_state and st.session_state['test_results']:
                for plugin_name, res in st.session_state['test_results'].items():
                    st.subheader(f"Results for {plugin_name}")
                    if res:
                        df = pd.DataFrame(res)
                        st.table(df)
                    else:
                        st.write("No tests found or parsing failed.")
            else:
                st.write(t("no_tests_run"))
        else:
            # Load and run only the selected plugin
            plugin_manager.run_plugin(st.session_state.selected_tab_id, config)
    except Exception as e:
        st.error(f"Application error: {str(e)}")


if __name__ == "__main__":
    main()
