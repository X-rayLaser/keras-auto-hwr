import json
import os
import api


@given(u'a pre-processor "{preprocessor_class}"')
def step_impl(context, preprocessor_class):
    context.data_provider = preprocessor_class


@given(u'a data set provider "{provider_class}"')
def step_impl(context, provider_class):
    context.preprocessor = provider_class


@when(u'I compile the data set with name "{name}" using at most {num_examples} examples')
def step_impl(context, name, num_examples):
    provider = context.data_provider
    preprocessor = context.preprocessor
    context.num_examples = int(num_examples)
    api.compile_data_set(data_provider=provider, preprocessor=preprocessor,
                         name=name, num_examples=num_examples)


@then(u'I confirm that the data set named "{name}" was successfully compiled')
def step_impl(context, name):
    json_string = api.data_set_info(name)
    d = json.loads(json_string)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    assert d['location'] == os.path.join(current_dir, 'compiled', name)
    assert d['preprocessor'] == context.preprocessor
    assert d['provider'] == context.provider
    assert d['number of examples'] == context.num_examples
    assert d['shape'] == ("?", "?", 4)
