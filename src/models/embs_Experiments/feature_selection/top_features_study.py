import os
import pandas as pd
import matplotlib.pyplot as plt


def comparison_models_sameDS(common_PATH, models2comp, DS, n_embs=4096, nFeat2selec=1000):

    df_ANOVA_ranking_models = pd.DataFrame(range(0,n_embs), columns=["idx_embs"])
    for model in models2comp:
        # Load features for each model:
        path_features_ANOVA = os.path.join(common_PATH, model, DS, "ANOVA-F", "statistics_ANOVA-F.csv")
        df_ANOVA_ranking = pd.read_csv(path_features_ANOVA, sep=";", header=0)
        df_ANOVA_ranking.columns = ["idx_embs", "F-stat_"+model, "p_values_"+model]

        df_ANOVA_ranking_models = df_ANOVA_ranking_models.merge(df_ANOVA_ranking, on="idx_embs")
    top_features_df = pd.DataFrame(range(0,nFeat2selec), columns=["idx_embs"])
    # Order dataframe by model features:
    for model in models2comp:
        ordered_df = df_ANOVA_ranking_models.sort_values(by=["F-stat_"+model], ascending=False)
        top_features_df["idx_embs_"+model] = ordered_df["idx_embs"].values[0:nFeat2selec]
        print(model, " - ", top_features_df["idx_embs_"+model].values)
    non_common_embeddings = (set(top_features_df[top_features_df.columns[1]].values).
                             difference(set(top_features_df[top_features_df.columns[2]].values)))
    print("Percentage of non-overlap: ", str(100*len(non_common_embeddings)/nFeat2selec))
    percentage_overlap = 100-(100*len(non_common_embeddings)/nFeat2selec)
    return percentage_overlap


def sameModelDifferentDS(common_PATH, model, DS2compare, n_embs=4096, nFeat2selec=1000):

    df_ANOVA_ranking_models = pd.DataFrame(range(0, n_embs), columns=["idx_embs"])
    for DS in DS2compare:
        # Load features for each model:
        path_features_ANOVA = os.path.join(common_PATH, model, DS, "ANOVA-F", "statistics_ANOVA-F.csv")
        df_ANOVA_ranking = pd.read_csv(path_features_ANOVA, sep=";", header=0)
        df_ANOVA_ranking.columns = ["idx_embs", "F-stat_" + DS, "p_values_" + DS]
        df_ANOVA_ranking_models = df_ANOVA_ranking_models.merge(df_ANOVA_ranking, on="idx_embs")

    top_features_df = pd.DataFrame(range(0, nFeat2selec), columns=["idx_embs"])
    # Order dataframe by model features:
    for DS in DS2compare:
        ordered_df = df_ANOVA_ranking_models.sort_values(by=["F-stat_" + DS], ascending=False)
        top_features_df["idx_embs_" + DS] = ordered_df["idx_embs"].values[0:nFeat2selec]
        print(DS, " - ", top_features_df["idx_embs_" + DS].values)
    non_common_embeddings = (set(top_features_df[top_features_df.columns[1]].values).
                             difference(set(top_features_df[top_features_df.columns[2]].values)))
    print("Percentage of non-overlap: ", str(100 * len(non_common_embeddings) / nFeat2selec))
    percentage_overlap = 100 - (100 * len(non_common_embeddings) / nFeat2selec)
    return percentage_overlap


def create_plot_overlaps(x, y, colors = [], label = [], title="", metric="", n_samples=[]):
    ax = plt.gca()
    n_y = 0
    for y_i in y:
        plt.plot(x, y_i, color=colors[n_y], marker='o', label=label[n_y])
        n_y+=1
    plt.grid(color='0.95')

    plt.legend()
    plt.xlabel("Number of Features")
    plt.ylabel(metric)
    plt.title(title)
    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    # parse input arguments
    common_PATH = "../../CBMS_MentalHealth_FeatureSelection/data/experiments/featureSelection"
    models2comp = ["meta-llama_Llama-2-7b-chat-hf","klyang_MentaLLaMA-chat-7B"]


    # CHECK OVERLAP OF FEATURES FOR SAME DATASET DIFFERENT MODELS:
    x = [10,50,100,200,300,400,500,600,700,800,900,1000]
    perc_overl_7cups = []
    for features in [10,50,100,200,300,400,500,600,700,800,900,1000]:
        perc_overl_7cups += [comparison_models_sameDS(common_PATH, models2comp, "7cups",n_embs=4096, nFeat2selec=features)]

    perc_overl_counselChat = []
    for features in [10,50,100,200,300,400,500,600,700,800,900,1000]:
        perc_overl_counselChat += [comparison_models_sameDS(common_PATH, models2comp, "counseilChat",n_embs=4096, nFeat2selec=features)]


    create_plot_overlaps(x, [perc_overl_7cups,perc_overl_counselChat], colors=['blue', 'orange'], metric= "Overlapped Dimensions (%)",
                         title="Percentage of overlap between \nLLama & Mental-Llama", label=["7cups", "counselChat"])


    #### CHECK OVERLAP OF FEATURES IN DATASETS / SAME MODEL:
    # x = [10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    # perc_overl = []
    # DS2compare = ["counseilChat", "7cups"]
    # model = "klyang_MentaLLaMA-chat-7B" #"meta-llama_Llama-2-7b-chat-hf"
    # for features in [10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
    #     perc_overl += [sameModelDifferentDS(common_PATH, model, DS2compare, n_embs=4096, nFeat2selec=features)]
    #
    # create_plot_overlaps(x, perc_overl, colors=['green'], metric="Overlapped Dimensions (%)",
    #                      title="Percentage of overlap between datasets for "+model)



