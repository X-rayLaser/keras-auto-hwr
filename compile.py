import api


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('data_provider', type=str)
    parser.add_argument('preprocessor_name', type=str)
    parser.add_argument('--ds_name', type=str, default='ds1')
    parser.add_argument('--num_examples', type=int, default=100)

    args = parser.parse_args()

    api.compile_data_set(data_provider=args.data_provider,
                         preprocessor_name=args.preprocessor_name,
                         name=args.ds_name, num_examples=args.num_examples)


# todo: IamSource should take path to iam data as argument
