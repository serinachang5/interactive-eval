import json
import requests
import os
import argparse
import pandas as pd


def fetch_data_by_hit(code, hitId):
    url = f'https://mturk-functions.azurewebsites.net/api/mturk-get-responses-by-hits?code={code}&hitId={hitId}'
    response = requests.get(url)
    return response.json()


if __name__ == '__main__':
    # parse the command line argument
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("hitId", help="HIT ID", type=str)
    args = arg_parser.parse_args()

    # get url_code from the last line of the .cosmosdb file
    lines = open(os.path.expanduser('.cosmosdb'), 'r').readlines()
    url_code = lines[-1].rstrip('\n')

    items = fetch_data_by_hit(url_code, args.hitId)    

    cols = ("hit_id", "assignment_id", "worker_id", "variable", "value")
    data = []
    print("\t".join(cols))

    for item in items:
        for key, value in item.items():
            if not (type(key) == str and key.startswith('_')):
                data.append(
                (item['hitId'], item['assignmentId'], item['workerId'], key, str(value))
                )
                print("\t".join((item['hitId'], item['assignmentId'] or "", item['workerId'] or "")), end = "\t")
                print(key, end = "\t")

                value = str(value).replace('\n', ' ').replace('\r', '').replace('\t', ' ')
                value_as_str = str(value.encode('ascii','replace').decode('utf-8'))
                print(value)
    
    df = pd.DataFrame(data, columns=cols)
    df.to_csv(f"./from_azure_{args.hitId}.csv", index=False)
