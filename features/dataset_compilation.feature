Feature: test compilation of raw data sets using different pre-processors

  Scenario: compiling a raw data set without doing any pre-processing
    Given a pre-processor "BasicPreprocessor"
    And a data set provider "DummyProvider"
    When I compile the data set with name "dummy" using at most 10 examples
    Then I confirm that the data set named "dummy" was successfully compiled
