import pandapower as pp
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
import utils

argparser = argparse.ArgumentParser()
argparser.add_argument("--network_name", help="network name", default="Srakovlje")
argparser.add_argument("--leto", help="year", default=2022)
argparser.add_argument("--season_train", help="season", default=['summer','midseason'])
argparser.add_argument("--season_test", help="season", default=['summer'])
argparser.add_argument("--weeks_sim_train", help="number of simulation weeks", default=350)
argparser.add_argument("--weeks_sim_test", help="number of simulation weeks", default=50)
argparser.add_argument("--PV_kW_comb", help="total number of installed PV", default=77)

argparser.add_argument("--save_type", help="data format to save the results", default="csv")
argparser.add_argument("--generation_save_file", help="path to the generation data", default="src/data/env_data/pv_gen")
argparser.add_argument("--load_save_file", help="path to the load data", default="src/data/env_data/loads")
argparser.add_argument("--generation_load_save_file", help="path to the load data", default="src/data/env_data/loads_and_generation")
args = argparser.parse_args()

# Branje profila transformatorja
#os.path.join('\src', 'inputs.yaml')#
TR_profil = pd.read_table("src/data/network_data/TR_profil.txt", delimiter='\t')

# Modeliranje matematičnega modela omrežja. Zaenkrat modeliramo omrežje za simulacijo simetričnih razmer
net, network_parameters, load_parameters, tr_parameters = utils.generate_network(args.network_name)
# save net to json
pp.to_json(net, f"src/data/env_data/{args.network_name}.json")

# start_date = pd.to_datetime(f"{args.leto}-01-01").date()

# season_idx = 0
# LD_diagram_vsi_tedni_train = pd.DataFrame()
# PV_diagram_vsi_tedni_train = pd.DataFrame()
# print("Generating train data...")
# for season in args.season_train:
#     # Generiranje vhodnih podatkov in simuliranje vseh tednov glede na število določeno v inputs.yaml datoteki
#     for i in tqdm(range(args.weeks_sim_train)):
#         LD_diagram = pd.DataFrame(np.zeros((672, len(load_parameters))))
#         PV_diagram = pd.DataFrame(np.zeros((672, len(load_parameters))))
#         # Generiranje odjema
#         LD_diagram = utils.generate_Loads(args.leto, season, load_parameters, start_date)
#         # Generiranje SE
#         PV_diagram = utils.load_diagram_PV(season, args.PV_kW_comb, load_parameters, start_date)
#         LD_diagram_vsi_tedni_train = pd.concat([LD_diagram_vsi_tedni_train, LD_diagram])
#         PV_diagram_vsi_tedni_train = pd.concat([PV_diagram_vsi_tedni_train, PV_diagram])
#         start_date = start_date + pd.DateOffset(days=7)


# season_idx = 0
# LD_diagram_vsi_tedni_test = pd.DataFrame()
# PV_diagram_vsi_tedni_test = pd.DataFrame()
# print("Generating test data...")
# for season in args.season_test:
#     # Generiranje vhodnih podatkov in simuliranje vseh tednov glede na število določeno v inputs.yaml datoteki
#     for i in tqdm(range(args.weeks_sim_test)):
#         LD_diagram = pd.DataFrame(np.zeros((672, len(load_parameters))))
#         PV_diagram = pd.DataFrame(np.zeros((672, len(load_parameters))))
#         # Generiranje odjema
#         LD_diagram = utils.generate_Loads(args.leto, season, load_parameters, start_date)
#         # Generiranje SE
#         PV_diagram = utils.load_diagram_PV(season, args.PV_kW_comb, load_parameters, start_date)
#         LD_diagram_vsi_tedni_test = pd.concat([LD_diagram_vsi_tedni_test, LD_diagram])
#         PV_diagram_vsi_tedni_test = pd.concat([PV_diagram_vsi_tedni_test, PV_diagram])
#         start_date = start_date + pd.DateOffset(days=7)

# if args.save_type == "csv":
#     # save to csv
#     PV_diagram_vsi_tedni_train.to_csv(args.generation_save_file + "_train.csv", index=True, header = True)
#     LD_diagram_vsi_tedni_train.to_csv(args.load_save_file + "_train.csv", index=True , header = True)

#     # calculate the difference between the generated and the actual load
#     diff_train = LD_diagram_vsi_tedni_train - PV_diagram_vsi_tedni_train
#     diff_train.to_csv(args.generation_load_save_file + "_train.csv", index=True, header = True)

#     PV_diagram_vsi_tedni_test.to_csv(args.generation_save_file + "_test.csv", index=True, header = True)
#     LD_diagram_vsi_tedni_test.to_csv(args.load_save_file + "_test.csv", index=True , header = True)

#     # calculate the difference between the generated and the actual load
#     diff_test = LD_diagram_vsi_tedni_test - PV_diagram_vsi_tedni_test
#     diff_test.to_csv(args.generation_load_save_file + "_test.csv", index=True, header = True)
# else:
#     print("Unknown export type...")


