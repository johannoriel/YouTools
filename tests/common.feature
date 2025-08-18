Feature: Simple Addition in Common Plugin

  Scenario: Add two positive numbers
    Given I have numbers 2 and 3
    When I add them
    Then the result should be 5

  Scenario: Add zero and a number
    Given I have numbers 0 and 5
    When I add them
    Then the result should be 5
