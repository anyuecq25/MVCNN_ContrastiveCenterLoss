from . import meter
import torch
import numpy as np
import scipy.io as sio


class RetrievalMAPMeter(meter.Meter):
    MAP = 0
    PR = 1

    def __init__(self, topk=900):
        self.topk = topk
        self.all_features = []
        self.all_lbs = []
        self.all_predict_lbs = []
        self.all_viewlist=[]
        self.dis_mat = None

        pass

    def reset(self):
        self.all_lbs.clear()
        self.all_features.clear()

    def add4(self, features, lbs, viewlist):
        self.all_features.append(features.cpu())
        self.all_lbs.append(lbs.cpu())
        self.all_viewlist.append(viewlist)

    def add5(self, features, lbs, viewlist,preds):
        self.all_features.append(features.cpu())
        self.all_lbs.append(lbs.cpu())
        self.all_viewlist.append(viewlist)
        self.all_predict_lbs.append(np.argmax(preds.cpu().numpy(), 1))
        #predict_lbl = self.all_predict_lbs.numpy()

    def add(self, features, lbs, viewlist):
        self.all_features.append(features.cpu())
        self.all_lbs.append(lbs.cpu())
        self.all_viewlist.append(viewlist)

    def add(self, features, lbs):
        self.all_features.append(features.cpu())
        self.all_lbs.append(lbs.cpu())


    def value(self, mode=MAP):
        if mode == self.MAP:
            return self.mAP()
        if mode == self.PR:
            return self.pr()
        raise NotImplementedError

    def mAP(self,feature_saved_mat_name='feature-mvcnn.mat'):
        fts = torch.cat(self.all_features).numpy()
        lbls = torch.cat(self.all_lbs).numpy()


        #sio.savemat('feature-mvcnn.mat', {'feature-mvcnn': fts})
        if(feature_saved_mat_name!='feature-mvcnn.mat') :
            if(len(lbls)>8987) and ('shrec14'  in feature_saved_mat_name):
                import h5py

                f = h5py.File(feature_saved_mat_name+'.h5', 'w')
                f['array'] = fts
                f['label'] = lbls
                #dt = h5py.special_dtype(vlen=str)
                #ds = f.create_dataset('test_dict', self.all_viewlist.shape, dtype=dt)
                #ds[:] = self.all_viewlist
                f.close()
                sio.savemat(feature_saved_mat_name+'_viewlist.mat',
                        {'viewlist': self.all_viewlist})  # add by

            else:
                #predict_lbl=self.all_predict_lbs.numpy()
                sio.savemat(feature_saved_mat_name, {'array': fts, 'label': lbls,'viewlist':self.all_viewlist,'predict_label':self.all_predict_lbs})  #add by qchen119 12/30/2019

        #return 0
        self.dis_mat = Eu_dis_mat_fast(np.mat(fts)) #calc distance matrix using features
        num = len(lbls)
        mAP = 0
        for i in range(num):
            scores = self.dis_mat[:, i]
            targets = (lbls == lbls[i]).astype(np.uint8)
            sortind = np.argsort(scores, 0)[:self.topk]
            truth = targets[sortind]
            sum = 0
            precision = []
            for j in range(self.topk):
                if truth[j]:
                    sum += 1
                    precision.append(sum * 1.0 / (j + 1))
            if len(precision) == 0:
                ap = 0
            else:
                for ii in range(len(precision)):
                    precision[ii] = max(precision[ii:])
                ap = np.array(precision).mean()
            mAP += ap
            # print(f'{i+1}/{num}\tap:{ap:.3f}\t')
        mAP = mAP / num
        return mAP

    def pr(self):
        fts = torch.cat(self.all_features).numpy()
        lbls = torch.cat(self.all_lbs).numpy()
        num = len(lbls)
        precisions = []
        recalls = []
        ans = []
        self.dis_mat = Eu_dis_mat_fast(np.mat(fts))  # calc distance matrix using features
        for i in range(num):
            scores = self.dis_mat[:, i]
            targets = (lbls == lbls[i]).astype(np.uint8)
            sortind = np.argsort(scores, 0)[:self.topk]
            truth = targets[sortind]
            tmp = 0
            sum = truth[:self.topk].sum()
            precision = []
            recall = []
            for j in range(self.topk):
                if truth[j]:
                    tmp += 1
                    # precision.append(sum/(j + 1))
                recall.append(tmp * 1.0 / sum)
                precision.append(tmp * 1.0 / (j + 1))
            precisions.append(precision)
            for j in range(len(precision)):
                precision[j] = max(precision[j:])
            recalls.append(recall)
            tmp = []
            for ii in range(11):
                min_des = 100
                val = 0
                for j in range(self.topk):
                    if abs(recall[j] - ii * 0.1) < min_des:
                        min_des = abs(recall[j] - ii * 0.1)
                        val = precision[j]
                tmp.append(val)
            #print('%d/%d' % (i + 1, num))
            ans.append(tmp)
        return np.array(ans).mean(0)


def Eu_dis_mat_fast(X):
    aa = np.sum(np.multiply(X, X), 1)
    ab = X * X.T
    D = aa + aa.T - 2 * ab
    D[D < 0] = 0
    D = np.sqrt(D)
    D = np.maximum(D, D.T)
    return D
