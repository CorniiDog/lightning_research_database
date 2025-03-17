import os
import database_parser
import logger

lightning_data_folder = "lylout_files"
data_extension = ".dat"

os.makedirs(lightning_data_folder, exist_ok=True) # Ensure that it exists

dat_file_paths = database_parser.get_dat_files_paths(lightning_data_folder, data_extension)

for file_path in dat_file_paths:
  if not logger.is_logged(file_path):
    print(file_path, "not appropriately added to the database. Adding...")
    database_parser.parse_lylout(file_path)
    logger.log_file(file_path) # Mark as logged and unmodified
  else:
    print(file_path, "was parsed and added to the database already")
