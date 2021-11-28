from data_generated import BasicDes, Autocorrelation, CTD, PseudoAAC, AAComposition, QuasiSequenceOrder
import pandas as pd
import numpy as np
import sys

"""
小批量结构化数据生成
"""
def cal_pep(peptides, sequence, results, type, output_path):
    peptides_descriptors = []
    count = 0
    temp = 0

    for peptide in peptides:
        if len(peptide) < 6:
            continue
        peptides_descriptor = {}
        peptide = str(peptide)
        AAC = AAComposition.CalculateAAComposition(peptide)
        DIP = AAComposition.CalculateDipeptideComposition(peptide)
        MBA = Autocorrelation.CalculateNormalizedMoreauBrotoAutoTotal(peptide, lamba=5)
        CCTD = CTD.CalculateCTD(peptide)
        QSO = QuasiSequenceOrder.GetSequenceOrderCouplingNumberTotal(peptide, maxlag=5)
        PAAC = PseudoAAC._GetPseudoAAC(peptide, lamda=5)
        APAAC = PseudoAAC.GetAPseudoAAC(peptide, lamda=5)
        Basic = BasicDes.cal_discriptors(peptide)
        peptides_descriptor.update(AAC)
        peptides_descriptor.update(DIP)
        peptides_descriptor.update(MBA)
        peptides_descriptor.update(CCTD)
        peptides_descriptor.update(QSO)
        peptides_descriptor.update(PAAC)
        peptides_descriptor.update(APAAC)
        peptides_descriptor.update(Basic)
        peptides_descriptors.append(peptides_descriptor)

        if count % 100 == 0:
            print("No.%d  Peptide: %s" % (count, peptide))
        count += 1
        # print(count)
    # write2csv(sequence,peptides_descriptors,results,"/home/xuyanchao/peptide_selection/datasets/all_data_with_negative_features.csv")
    write2csv(sequence, peptides_descriptors, results, type, output_path)


def write2csv(sequence, input_data, result, type, output_path):
    # print(input_data[0])
    df = pd.DataFrame(input_data)
    output_csv = pd.concat([sequence, df, result, type], axis=1)
    output_csv.to_csv(output_path, encoding="utf8")


if __name__ == "__main__":
    # file = "/home/xuyanchao/peptide_selection/datasets/all_data_with_negative_features.csv"
    file = "./classify_test_sample.csv"
    data = pd.read_csv(file, encoding="utf-8")
    sequence = data["sequence"]
    peptides = sequence.values.copy().tolist()
    result = data["MIC"]
    type = data["type"]
    output_path = "./regression_feature.csv"
    cal_pep(peptides, sequence, result, type, output_path)
