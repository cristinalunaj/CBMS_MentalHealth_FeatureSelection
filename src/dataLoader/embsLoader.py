"""
	author: Cristina Luna,

"""
import os
import pandas as pd
import numpy as np
import fnmatch




def load_labels_CounseilChat(path_labels):
    df_labels = pd.read_csv(path_labels, sep=";", header=0)
    return df_labels




class EmbsLoaderCounselChat():
    def __init__(self, path_embs, path_labels, col_idx="idx_set", col_setName = "setName", normalize=False, filterBycol={}):
        print("Loading labels... ")
        self.labels_df = load_labels_CounseilChat(path_labels)
        self.col_idx = col_idx
        self.set2load = filterBycol[col_setName] if (col_setName in filterBycol.keys()) else "Complete"
        self.idxInSet = self.get_filtered_Idx(filterBycol) #self.get_idx_ofSet()
        self.path_embs = path_embs
        self.normalize = normalize




    def get_filtered_Idx(self, filterByCols):
        labels_df_aux = self.labels_df
        for k in filterByCols:
            print("Filtering data by: ", str(k), ":", str(filterByCols[k]), "...")
            labels_df_aux = labels_df_aux.loc[labels_df_aux[k] == filterByCols[k]]
        setIdx = labels_df_aux[self.col_idx]
        print("Final length: ", str(len(setIdx)))
        return setIdx



    def load_labels(self, path_labels):
        return pd.read_csv(path_labels, sep=";", header=0)


    def load_embs(self, n_embs=768):
        col_names = ['emb{}'.format(i) for i in range(n_embs)]
        complete_embs = pd.DataFrame([], columns=[self.col_idx] + col_names)

        if(self.normalize):
            path_avg_complete_embs = os.path.join(self.path_embs, "NORMALIZED_"+self.set2load + "_avg_complete_embs.csv")
        else:
            path_avg_complete_embs = os.path.join(self.path_embs, self.set2load+"_avg_complete_embs.csv")

        if(os.path.exists(path_avg_complete_embs)):
            complete_embs = pd.read_csv(path_avg_complete_embs, sep=";", header=0)
            complete_embs = complete_embs[[self.col_idx] + col_names]
        else:
            try:
                for idx in self.idxInSet:
                    # list all the numpy arrays and order:
                    path_emb_npy = os.path.join(self.path_embs, str(idx))
                    npys = fnmatch.filter(os.listdir(path_emb_npy), "*.npy")
                    avg_embs_df_parts = pd.DataFrame([], columns=col_names)
                    for part in sorted(npys):
                        with open(os.path.join(path_emb_npy, part), 'rb') as f:
                            a = np.load(f)
                        avg_emb = a.mean(axis=0)
                        aux_df = pd.DataFrame([list(avg_emb.reshape(-1))], columns=col_names)

                        avg_embs_df_parts = pd.concat([aux_df, avg_embs_df_parts.loc[:]]).reset_index(drop=True)
                    # Calculate average
                    avg_embs_df_parts = pd.DataFrame([avg_embs_df_parts.mean().values], columns=col_names)
                    avg_embs_df_parts[self.col_idx] = idx
                    # Append
                    complete_embs = pd.concat([avg_embs_df_parts, complete_embs.loc[:]]).reset_index(drop=True)
            except EOFError:
                print(path_emb_npy)
                print("to do")
            # Save complete embs:
            complete_embs.to_csv(path_avg_complete_embs, sep=";", header=True, index=False)
        print("LOADED: ", len(complete_embs), " embeddings")
        return complete_embs, col_names


    def load_idx_embs(self, idx, n_embs=768):
        # list all the numpy arrays and order:
        col_names = ['emb{}'.format(i) for i in range(n_embs)]
        path_emb_npy = os.path.join(self.path_embs, str(idx))
        npys = fnmatch.filter(os.listdir(path_emb_npy), "*.npy")
        avg_embs_df_parts = pd.DataFrame([], columns=col_names)
        for part in sorted(npys):
            with open(os.path.join(path_emb_npy, part), 'rb') as f:
                a = np.load(f)
            avg_emb = a.mean(axis=0)
            aux_df = pd.DataFrame([list(avg_emb.reshape(-1))], columns=col_names)

            avg_embs_df_parts = pd.concat([aux_df, avg_embs_df_parts.loc[:]]).reset_index(drop=True)
        # Calculate average
        avg_embs_df_parts = pd.DataFrame([avg_embs_df_parts.mean().values], columns=col_names)
        avg_embs_df_parts[self.col_idx] = idx
        # Append
        return avg_embs_df_parts


    def get_labels(self, labels_cols):
        return self.labels_df[self.labels_df[self.col_idx].isin(self.idxInSet)][[self.col_idx]+labels_cols]


