

aws s3 sync  /vol/projects/jnourisa/datasets/bulk/ s3://openproblems-data/resources/grn/temp/datasets/bulk/ --delete
aws s3 sync  /vol/projects/jnourisa/output/grns s3://openproblems-data/resources/grn/temp/output/grns --delete

# base_dir='/Users/jno24/Documents/projs/ongoing/ciim/base_folder/'
# aws s3 sync s3://openproblems-data/resources/grn/temp/datasets/bulk/ ${base_dir}/${folder}/ 
