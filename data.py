import json
import requests
import os


def loadAllId():
    all_id = []

    for i in range(0, 90001, 5000):
        my_query = {
            "query": {
                "type": "group",
                "nodes": [
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "attribute": "exptl_crystal_grow.temp",
                            "operator": "greater_or_equal",
                            "value": 273,
                            "negation": False
                        }
                    },
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "attribute": "exptl_crystal_grow.temp",
                            "operator": "less_or_equal",
                            "value": 300,
                            "negation": False
                        }
                    },
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "attribute": "exptl_crystal_grow.pH",
                            "operator": "greater_or_equal",
                            "value": 5,
                            "negation": False
                        }
                    },
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "attribute": "exptl_crystal_grow.pH",
                            "operator": "less_or_equal",
                            "value": 8,
                            "negation": False
                        }
                    },
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "attribute": "entity_poly.rcsb_entity_polymer_type",
                            "operator": "exact_match",
                            "value": "Protein",
                            "negation": False
                        }
                    }
                ],
                "logical_operator": "and",
                "label": "text"
            },
            "return_type": "entry",
            "request_options": {
                "paginate": {
                    "rows": 5000,
                    "start": i
                },
                "results_content_type": ["experimental"],
                "sort": [
                    {
                        "sort_by": "score",
                        "direction": "desc"
                    }
                ],
                "scoring_strategy": "combined"
            }
        }
        my_query = json.dumps(my_query)
        data = requests.get(f"https://search.rcsb.org/rcsbsearch/v2/query?json={my_query}")
        results = data.json()
        for each in results['result_set']:
            all_id.append(each['identifier'])
    return all_id



def getSequence():
    prevVal = False
    ## Default do not change
    query = '''
    query ($id: String!) {
        entry(entry_id: $id) {
            polymer_entities{
                entity_poly {
                    pdbx_seq_one_letter_code_can
                }
            }
        }
    }
    '''
    urlSequence = "https://data.rcsb.org/graphql"
    headers = {
        "Content-Type": "application/json"
    }
    ##################################################
    with open('src/dataCollection/id.txt', 'r') as all_id:
        for id in all_id:
            id = id.strip() # remove \n
            if id == "3CY1":
                prevVal = True
            if prevVal == True:
                url = f"https://files.rcsb.org/download/{id}.pdb"
                folder_path = os.path.join('src/dataCollection/', id)
                # print(folder_path)

                ## Download pdb file and put to local
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                try:
                    response = requests.get(url)
                    if response.status_code == 200:
                        file_path = os.path.join(folder_path, f"{id}.pdb")
                        with open(file_path, 'wb') as file:
                            file.write(response.content)
                        
                        print(f"write {id}.pdb successfully!")
                    else:
                        print(f"Failed to download {id}.pdb")
                except Exception as e:
                    print("error message: ", str(e))

                ## get sequence and write to local
                variables = {
                    "id": f"{id}"
                }
                payload = {
                    "query": query,
                    "variables": variables
                }
                response = requests.post(urlSequence, json=payload, headers=headers)
                if response.status_code == 200:
                    data = response.json()
                    file_path = os.path.join(folder_path, f"{id}.txt")
                    with open(file_path, 'w') as file:
                        file.write(data['data']['entry']['polymer_entities'][0]['entity_poly']['pdbx_seq_one_letter_code_can'])
                        print(f"write {id}.txt successfully!")
                else:
                    print(f"write {id}.txt failed!")

# getSequence()