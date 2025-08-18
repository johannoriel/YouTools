from pytest_bdd import given, when, then, parsers
from plugins.common import CommonPlugin
from app import PluginManager

@given(parsers.parse('I have numbers {a:d} and {b:d}'), target_fixture='numbers')
def numbers(a, b):
    return {'a': a, 'b': b}

@when('I add them')
def add(numbers):
    mock_manager = PluginManager({'common': {}})
    plugin = CommonPlugin("common", mock_manager)
    numbers['result'] = plugin.add_numbers(numbers['a'], numbers['b'])

@then(parsers.parse('the result should be {result:d}'))
def check_result(numbers, result):
    assert numbers['result'] == result
