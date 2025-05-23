import yaml
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    env_type = config['env']['type']

    if env_type == 'unity':
        from train import train_a2c
        train_a2c.train(config)
    else:
        raise ValueError(f"Unsupported environment type: {env_type}")

if __name__ == '__main__':
    main()
