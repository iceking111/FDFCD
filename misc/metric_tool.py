import numpy as np

"""

这段代码定义了几个用于计算和跟踪机器学习模型性能指标的类和函数，特别是在分类任务中常用的混淆矩阵相关的指标。以下是每个类和函数的简要说明：

AverageMeter 类
用于计算和存储平均值和当前值。
initialize 方法用于初始化变量。
update 方法用于更新当前值和权重。
add 方法用于累加值和权重，并重新计算平均值。
value 和 average 方法分别返回当前值和平均值。
get_scores 方法用于从累积的和中获取分数，但这个方法在 AverageMeter 类中没有具体实现。


ConfuseMatrixMeter 类
继承自 AverageMeter 类，添加了针对混淆矩阵的具体功能。
update_cm 方法接收预测值和真实标签，并更新混淆矩阵，同时计算当前的 F1 分数。
get_scores 方法返回混淆矩阵的详细分数，包括准确度、召回率、精确度和 F1 分数。
函数
harmonic_mean：计算一组数值的调和平均数。
cm2F1：根据混淆矩阵计算 F1 分数。
cm2score：根据混淆矩阵计算多个分数，包括准确度、平均交并比（mIoU）、F1 分数等，并将它们组织成字典返回。
get_confuse_matrix：计算一组预测的混淆矩阵。
get_mIoU：计算并返回平均交并比（mIoU）。
混淆矩阵
混淆矩阵是一个表格，用于描述分类模型的性能。它显示了每个类别的真实标签和模型预测标签之间的关系。矩阵的行表示真实标签，列表示预测标签。

F1 分数
F1 分数是精确度和召回率的调和平均数，用于衡量模型的准确性和完整性的平衡。它是评价模型性能的一个指标，特别是在类别不平衡的情况下。

mIoU
平均交并比（mean Intersection over Union，mIoU）是多个类别的交并比的平均值，用于评估多分类模型的性能。它衡量的是预测的类别标签和真实标签之间的一致性。

这段代码可以用于跟踪和评估分类模型的性能，特别是在需要详细分析每个类别的性能时。通过这些指标，研究人员和开发人员可以了解模型在哪些类别上表现良好，在哪些类别上需要改进。


"""
###################       metrics      ###################
# 定义了一个名为AverageMeter的类，用于计算和存储数值的平均值和当前值。一般在深度学习中用于跟踪训练过程中的指标，比如损失值和准确率等
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False# 是否已经初始化
        self.val = None# 当前值
        self.avg = None# 平均值
        self.sum = None# 所有值的总和
        self.count = None# 权重的总和

    def initialize(self, val, weight): # 初始化操作 接受一个当前值和权重
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):# 更新当前值和平均值，若已经初始化，调用add方法，否则先初始化
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):# 接受一个val和weight，然后更新几个值
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):# 返回当前值
        return self.val

    def average(self):# 返回平均值
        return self.avg

    def get_scores(self):#
        scores_dict = cm2score(self.sum)
        return scores_dict

    def clear(self):# 将初始化设置为False，从而重置AverageMeter对象
        self.initialized = False


###################      cm metrics      ###################
# 定义了一个混淆矩阵的类 用于计算和存储混淆矩阵的平均值和当前值 提供更新混淆矩阵以及获取性能指标的方法
class ConfuseMatrixMeter(AverageMeter):
    """Computes and stores the average and current value"""
    def __init__(self, n_class):
        super(ConfuseMatrixMeter, self).__init__()
        self.n_class = n_class
    # 定义了一个名为 update_cm 的方法，用于更新混淆矩阵。它接受三个参数：pr（预测标签），gt（真实标签），以及一个可选参数 weight，默认值为1。
    def update_cm(self, pr, gt, weight=1):
        """获得当前混淆矩阵，并计算当前F1得分，并更新混淆矩阵"""
        # 调用 get_confuse_matrix 函数来获取当前的混淆矩阵。这个函数需要类别数量 num_classes，真实标签 label_gts 和预测标签 label_preds。
        val = get_confuse_matrix(num_classes=self.n_class, label_gts=gt, label_preds=pr)
        # 传入当前的混淆矩阵 val 和权重 weight，以更新平均混淆矩阵。
        self.update(val, weight)
        # 调用 cm2F1 函数来计算当前混淆矩阵的F1分数。
        current_score = cm2F1(val)
        # 返回计算得到的当前F1分数。
        return current_score
    # 定义了一个名为 get_scores 的方法，用于获取性能指标。
    def get_scores(self):
        scores_dict = cm2score(self.sum)
        return scores_dict


# 定义了一个名为 harmonic_mean 的函数，其目的是计算传入列表 xs 的调和平均值。
def harmonic_mean(xs):
    # 计算调和平均值。调和平均值的计算公式是如下
    harmonic_mean = len(xs) / sum((x+1e-6)**-1 for x in xs)
    return harmonic_mean

# 混淆矩阵用处1
# 定义了一个名为 cm2F1 的函数，其目的是根据提供的混淆矩阵计算平均F1分数。与下面的一样的
def cm2F1(confusion_matrix):
    hist = confusion_matrix
    n_class = hist.shape[0]
    tp = np.diag(hist)
    sum_a1 = hist.sum(axis=1)
    sum_a0 = hist.sum(axis=0)
    # ---------------------------------------------------------------------- #
    # 1. Accuracy & Class Accuracy
    # ---------------------------------------------------------------------- #
    acc = tp.sum() / (hist.sum() + np.finfo(np.float32).eps)

    # recall
    recall = tp / (sum_a1 + np.finfo(np.float32).eps)
    # acc_cls = np.nanmean(recall)

    # precision
    precision = tp / (sum_a0 + np.finfo(np.float32).eps)

    # F1 score
    F1 = 2 * recall * precision / (recall + precision + np.finfo(np.float32).eps)
    mean_F1 = np.nanmean(F1)
    return mean_F1

# 混淆矩阵用处2
# 计算并返回基于混淆矩阵的多种性能指标。传入的混淆矩阵是一个形状为(n_class,n_class)的二维数组hist
def cm2score(confusion_matrix):
    # 将输入的混淆矩阵赋值给hist数组变量
    hist = confusion_matrix
    # 获取混淆矩阵的行数，也就是总类别的个数
    n_class = hist.shape[0]
    # 计算对角线上的元素，这些元素代表每个类别正确预测的样本数量，存在tp中
    tp = np.diag(hist)
    # 按照列(每个类别)对混淆矩阵进行求和
    sum_a1 = hist.sum(axis=1)
    # 按照行(每个类别的预测)对混淆矩阵进行求和
    sum_a0 = hist.sum(axis=0)
    # ---------------------------------------------------------------------- #
    # 1. Accuracy & Class Accuracy
    # ---------------------------------------------------------------------- #
    # 计算准确率 将对角线上元素(正确预测的样本数)相加 除以混淆矩阵所有元素和(总样本数)，为了避免除0，因此使用np.finfo(np.float32).eps
    acc = tp.sum() / (hist.sum() + np.finfo(np.float32).eps)

    # 计算召回率 对角线上元素(正确预测的样本数)除以每类实际样本数
    recall = tp / (sum_a1 + np.finfo(np.float32).eps)# 计算召回率
    # acc_cls = np.nanmean(recall)

    # 计算精准度 对角线上元素(正确预测的样本数)除以每类预测的样本数
    precision = tp / (sum_a0 + np.finfo(np.float32).eps)# 计算精准度

    # 计算F1分数 召回率和精准率的调和平均值 下面是计算公式
    F1 = 2*recall * precision / (recall + precision + np.finfo(np.float32).eps) # 计算F1分数
    # 计算F1的平均值，用np.nanmean忽略数组中的NaN值
    mean_F1 = np.nanmean(F1)
    # ---------------------------------------------------------------------- #
    # 2. Frequency weighted Accuracy & Mean IoU
    # ---------------------------------------------------------------------- #
    # 计算交并比IoU 对角线上元素(正确预测的样本数)除以对角线元素与按行求和的差(即正确预测数加上实际样本数减去预测的样本数)
    iu = tp / (sum_a1 + hist.sum(axis=0) - tp + np.finfo(np.float32).eps)# 计算交互比 Iou
    mean_iu = np.nanmean(iu)# 计算平均值

    # 计算每个类别的出现频率 类别的实际样本数除以总体样本数
    freq = sum_a1 / (hist.sum() + np.finfo(np.float32).eps)
    # 计算频率加权准确率 仅考虑频率大于0的类别，计算频率与IoU的乘积之和
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()

    # 为每一个类别创建一个包含交并比的字典，键为类别名称，值是对应的IoU值
    cls_iou = dict(zip(['iou_'+str(i) for i in range(n_class)], iu))
    # 为每个类别创建一个包含精确率的字典，键为类别名称，值是对应的精确率值。
    cls_precision = dict(zip(['precision_'+str(i) for i in range(n_class)], precision))
    # 为每个类别创建一个包含召回率的字典，键为类别名称，值是对应的召回率值。
    cls_recall = dict(zip(['recall_'+str(i) for i in range(n_class)], recall))
    # 为每个类别创建一个包含F1分数的字典，键为类别名称，值是对应的F1分数值。
    cls_F1 = dict(zip(['F1_'+str(i) for i in range(n_class)], F1))
    # 创建一个包含总体性能指标的字典：准确率、平均IoU和平均F1分数。
    score_dict = {'acc': acc, 'miou': mean_iu, 'mf1':mean_F1}
    # 将类别级别的性能指标添加到总体性能指标字典中。
    score_dict.update(cls_iou)
    score_dict.update(cls_F1)
    score_dict.update(cls_precision)
    score_dict.update(cls_recall)
    # 返回包含所有计算出的性能指标的字典
    return score_dict

# 定义了一个名为 get_confuse_matrix 的函数，该函数接收三个参数：num_classes（类别总数），label_gts（真实标签数组），label_preds（预测标签数组）。
def get_confuse_matrix(num_classes, label_gts, label_preds):
    """计算一组预测的混淆矩阵"""
    # 定义了一个内部函数 __fast_hist，用于收集混淆矩阵的值。参数是 label_gt（真实标签）和 label_pred（预测标签）。
    def __fast_hist(label_gt, label_pred):
        """
        Collect values for Confusion Matrix
        For reference, please see: https://en.wikipedia.org/wiki/Confusion_matrix
        :param label_gt: <np.array> ground-truth
        :param label_pred: <np.array> prediction
        :return: <np.ndarray> values for confusion matrix
        """
        # 创建一个布尔掩码，用于筛选出有效的标签索引（即在0到 num_classes-1 之间的标签）。
        mask = (label_gt >= 0) & (label_gt < num_classes)
        # 使用 np.bincount 函数计算混淆矩阵的值。计算方法是将每个标签对的索引相加，然后对这些索引进行计数。
        # num_classes * label_gt[mask].astype(int) + label_pred[mask] 计算出标签对的索引
        # minlength=num_classes**2 确保数组长度足够
        # reshape(num_classes, num_classes) 将一维数组重塑为二维数组，形成混淆矩阵。
        hist = np.bincount(num_classes * label_gt[mask].astype(int) + label_pred[mask],
                           minlength=num_classes**2).reshape(num_classes, num_classes)
        return hist
    # 初始化一个大小为 (num_classes, num_classes) 的零矩阵，用于存储最终的混淆矩阵。
    confusion_matrix = np.zeros((num_classes, num_classes))
    for lt, lp in zip(label_gts, label_preds):
        confusion_matrix += __fast_hist(lt.flatten(), lp.flatten())
    # 返回最终的混淆矩阵
    return confusion_matrix

# 定义了一个名为 get_mIoU 的函数，接收三个参数：num_classes（类别总数），label_gts（真实标签数组），label_preds（预测标签数组）。
def get_mIoU(num_classes, label_gts, label_preds):
    # 调用 get_confuse_matrix 函数计算混淆矩阵，并将其存储在变量 confusion_matrix 中。
    confusion_matrix = get_confuse_matrix(num_classes, label_gts, label_preds)
    # 调用 cm2score 函数，传入混淆矩阵，计算并获取一个包含多种性能指标的字典 score_dict。
    score_dict = cm2score(confusion_matrix)
    # 从 score_dict 字典中获取平均交并比（mean Intersection over Union，简称 mIoU），并将其作为函数的返回值。
    return score_dict['miou']
