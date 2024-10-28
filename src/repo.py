def jaccard_cosine_nets():
    nets_equal_size = {key: net.sort_values(by='weight', ascending=False, key=abs)[:min_net_size] for key, net in nets.items()} # keep top 100 links
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5), sharey=True)
    cousine_matrix, fig = cosine_similarity(nets_equal_size, col_name='link', weight_col='weight', figsize=(5.5, 4.5), ax=ax1, title=f'cousine similarity')
    jaccard_matrix, fig = jaccard_similarity(nets_equal_size, col_name='link', figsize=(5.5, 4.5), ax=ax2, title=f'jaccard similarity')
    plt.tight_layout()

    
def extract_geneset_data(coexp_all):
    genesets = get_genesets()
    coexp_geneset_dfs = {}
    mean_coexp_geneset_dict = {}
    results_list = []
    for name, geneset in genesets.items():
        geneset_links = ['_'.join(sorted(combination)) for combination in list(itertools.combinations(geneset,2))]
        geneset_links_present = np.intersect1d(geneset_links, coexp_all.columns)
        print(f"{name}: from {len(geneset_links)} links, present: {len(geneset_links_present)}")
        coexp_genesets = coexp_all[geneset_links_present]
        coexp_geneset_dfs[name] = coexp_genesets
        mean_coexp_geneset_dict[name] = coexp_genesets.abs().mean(axis=1)

        for index, row in coexp_genesets.iterrows():
            for link, sub_row in row.to_frame().iterrows():
                results_list.append({
                                    'geneset': name,
                                    'age': index,
                                    'link': link,
                                    'value': sub_row[0]
                                })
    coexp_geneset_melted = pd.DataFrame(results_list)
    coexp_geneset_mean = pd.DataFrame(mean_coexp_geneset_dict)
    return coexp_geneset_melted, coexp_geneset_mean


def process_grns():
    genesets = get_genesets()
    all_genesets = np.unique(np.concatenate(list(genesets.values())))

    par['models_dir'] = 'output/grn_models/pearson_corr_scgen_pearson'
    for i, model in enumerate(par['models']):
        net = pd.read_csv(f"{par['models_dir']}/{model}.csv", index_col=0) 
        print(model, 'size :', len(net))
        net['link'] = net['source'].astype(str) + '_' + net['target'].astype(str)
        if True: # subset to prior genesets
            geneset_links = ['_'.join(sorted(combination)) for combination in list(itertools.combinations(all_genesets,2))]
            net = net[net.link.isin(geneset_links)]
        # - make the links as column and weight as values
        coexp_df = pd.DataFrame(net.weight.values.reshape(1, len(net)), columns=net.link.values, index=[model])
        if i == 0:
            coexp_all = coexp_df
        else:
            coexp_all = pd.concat([coexp_all, coexp_df], axis=0).fillna(0)
        if i == 0:
            min_net_size = len(net)
        else:
            min_net_size = min([min_net_size, len(net)])   
        print(coexp_all.shape) 
    coexp_all.to_csv('output/full/coexp_all.csv')
    # print([len(net) for k, net in nets.items()])
    print('ratio of zeros:', (coexp_all==0).sum().sum()/coexp_all.size)