import json
import requests
import os
import subprocess

## Get all PDB IDs
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


# Get fasta and pdb files
def getSequence():
    counter = 0
    prevVal = False
    all_id = loadAllId()
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
    # For every PDB ID, download pdb file and fasta file -> then generate MSA
    for id in all_id:
        id = id.strip() # remove \n
        if id == "1B14":
            prevVal = True
        if prevVal:
            url = f"https://files.rcsb.org/download/{id}.pdb"
            pdb_path = 'model/targetPDB/'
            seq_path = 'model/input_seqs'
            
            # Download pdb file
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    file_path = os.path.join(pdb_path, f"{id}.pdb")
                    with open(file_path, 'wb') as file:
                        file.write(response.content)
                    
                    print(f"write {id}.pdb successfully!")
                else:
                    print(f"Failed to download {id}.pdb")
            except Exception as e:
                print("error message: ", str(e))

            ## get sequence and write fasta file
            fasta_url = f"https://rcsb.org/fasta/entry/{id}"
            variables = {
                "id": f"{id}"
            }
            payload = {
                "query": query,
                "variables": variables
            }
            response = requests.post(fasta_url, json=payload, headers=headers)
            if response.status_code == 200:
                file_path = os.path.join(seq_path, f"{id}.fasta")
                with open(file_path, 'w') as file:
                    file.write(response.text)
                    print(f"write {id}.fasta successfully!")
            else:
                print(f"write {id}.fasta failed!")

            # Generate MSA
            processMsa(id)

            counter += 1
            if counter == 10:
                print(f"{counter} MSA/PDB files/fasta files processed")

## Get MSA
def processMsa(file):
    input_fasta = f"model/input_seqs/{file}.fasta"  # sequence
    output_a3m = f"model/msa_raw/{file}.a3m"    # msa
    database_path = "/Users/hahayes/Desktop/Capstone/hh-suite/databases/uniclust30_2016_09/uniclust30_2016_09" # database

    # HHblits command
    hhblits_command = [
        "hhblits", 
        "cpu", "4",
        "-i", input_fasta,          # sequence
        "-d", database_path,        # database
        "-oa3m", output_a3m,        # msa
        "-norealign",               # norealign
        "-n", "3"                   # iteration
    ]

    # run command
    try:
        subprocess.run(hhblits_command, check=True)
        print(f"MSA successfully generated: {file}")
    except subprocess.CalledProcessError as e:
        print(f"HHblits error: {e}")


getSequence()