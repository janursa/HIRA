folder='datasets/metacell/'

base_dir='/vol/projects/jnourisa/'
aws s3 sync  ${base_dir}/${folder}/ s3://openproblems-data/resources/grn/temp/${folder}/ 

# base_dir='/Users/jno24/Documents/projs/ongoing/ciim/base_folder/'
# aws s3 sync s3://openproblems-data/resources/grn/temp/datasets/bulk/ ${base_dir}/${folder}/ 
