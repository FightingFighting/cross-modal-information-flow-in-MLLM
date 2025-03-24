
# Scienfitic packages
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

# Visuals

import seaborn as sns

import os

from matplotlib.ticker import FuncFormatter


import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'


def read_csv(path):
    tmp = pd.read_csv(path)

    tmp['block_desc'] = tmp['block_desc'].str.replace("Image Central Object", "Related Image Patches")
    tmp['block_desc'] = tmp['block_desc'].str.replace("Image Objects", "Related Image Patches")
    tmp['block_desc'] = tmp['block_desc'].str.replace("Image Without Central Object", "Other Image Patches")
    tmp['block_desc'] = tmp['block_desc'].str.replace("Other Image Patches with pad", "Other Image Patches")
    tmp['block_desc'] = tmp['block_desc'].str.replace("Image Without Objects", "Other Image Patches")
    tmp['block_desc'] = tmp['block_desc'].str.replace("last", "Last")
    return tmp



def generate_legend(data, operation_path, model_name, y="relative diff first", palette=None, save_fold=None):
    data["block_desc"] = data["block_desc"].str.replace("->", r"$\\nrightarrow$", regex=True)

    sns.set(context="notebook")

    sns.set_theme(style='whitegrid')
    plt.figure(figsize=(4, 4))
    ax = sns.lineplot(data, x="layer", y=y,
                      hue="block_desc",
                      style="block_desc",
                      dashes=False,
                      palette=palette, linewidth=3)


    fig_legend = plt.figure(figsize =(4,1))
    plt.tight_layout()

    ax_legend = fig_legend.add_subplot(111)


    ax_legend.legend(*ax.get_legend_handles_labels(), ncol=3, fontsize=22, frameon=False,  alignment="center")
    ax_legend.axis("off")
    # plt.subplots_adjust(right=1,left=0, bottom=0, top=1)
    result_path = f"{operation_path}/figures/{model_name}/{save_fold}"
    os.makedirs(result_path, exist_ok=True)
    save_file = os.path.join(result_path, f"legend.pdf")
    fig_legend.savefig(save_file, bbox_inches="tight")




def generate_plot(data, save_file, operation_path, model_name, y="relative diff first", layer_num=None, palette=None, save_fold=None):
    data["block_desc"] = data["block_desc"].str.replace("->", r"$\\nrightarrow$", regex=True)
    sns.set(context="notebook")

    sns.set_theme(style='whitegrid')
    plt.figure(figsize=(4, 4))
    ax = sns.lineplot(data, x="layer", y=y,
                      hue="block_desc",
                      style="block_desc",
                      dashes=False,
                      palette=palette, linewidth=3)
    ax.set_xlabel("Layer", fontsize=19)
    ax.set_ylabel("Change in probability (%)", fontsize=19)
    ax.set_xlim(0, layer_num)

    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: '{:,.0f}'.format(x)))

    y_ticks = ax.get_yticks()

    new_y_ticks = [tick for tick in y_ticks if tick <= 0]

    ax.set_yticks(new_y_ticks)



    ax.tick_params(axis='x', labelsize=12,  direction="in")
    ax.tick_params(axis='y', labelsize=12,  direction="in")

    plt.legend().remove()
    plt.tight_layout()

    filename = save_file.split("/")[-1]
    result_path = f"{operation_path}/figures/{model_name}/{save_fold}"
    os.makedirs(result_path, exist_ok=True)

    save_file = os.path.join(result_path, filename)
    plt.subplots_adjust(right=0.98,left=0.16, bottom=0.135, top=0.995)
    plt.savefig(save_file)



# palette = sns.color_palette("Accent")


palette_set1 = sns.color_palette("Set1") #9 colors
# palette = palette[2:5] + palette[7:]+ palette[5:6]+ palette[0:2]


hex_colors=["#a7c957", "#ffcc00", "#f77f00", "#fe7f2d", "#70d6ff", "#ff70a6"]
palette = sns.color_palette(hex_colors)


palette = palette[:2]+palette_set1[0:1]+palette_set1[1:2]+palette_set1[2:3]+palette_set1[7:8]






task1="GQA_val_correct_question_with_choose_ChooseAttr.csv"
task2="GQA_val_correct_question_with_positionQuery_QueryAttr.csv"
task3="GQA_val_correct_question_with_existThatOr_LogicalObj.csv"
task4="GQA_val_correct_question_with_twoCommon_CompareAttr.csv"
task5="GQA_val_correct_question_with_relChooser_ChooseRel.csv"
task6="GQA_val_correct_question_with_categoryThatThisChoose_objThisChoose_ChooseCat.csv"


def merge(model_name, operation_path, window):
    # ######################### to last #########################
    for ind, task_name in enumerate([task1, task2, task3, task4, task5, task6]):
        data1 = read_csv(
            path=operation_path + f"information_flow/{model_name}/{task_name.split('_')[-1].split('.csv')[0]}/val/Question___Last/{task_name.split('.csv')[0]}_window{window}_Question___Last.csv")
        data2 = read_csv(
            path=operation_path + f"information_flow/{model_name}/{task_name.split('_')[-1].split('.csv')[0]}/val/Image___Last/{task_name.split('.csv')[0]}_window{window}_Image___Last.csv")
        data3 = read_csv(
            path=operation_path + f"information_flow/{model_name}/{task_name.split('_')[-1].split('.csv')[0]}/val/Last___Last/{task_name.split('.csv')[0]}_window{window}_Last___Last.csv")
        # combine
        data_combined = pd.concat([data1, data2, data3])

        if ind == 0:
            generate_legend(data_combined,
                            operation_path,
                            model_name=model_name,
                            y="relative diff first",
                            palette=palette,
                            save_fold="toLast")
        generate_plot(data_combined,
                      save_file=operation_path + f"information_flow/{model_name}/{task_name.split('_')[-1].split('.csv')[0]}/val/{model_name}_{task_name.split('.csv')[0]}_window{window}_ToLast_firstgen.pdf",
                      operation_path=operation_path,
                      model_name=model_name,
                      y="relative diff first",
                      layer_num=31,
                      palette=palette,
                      save_fold="toLast")

        print(f"finish: {task_name}")

    ######################### Image to Question ##########################################
    path1 = operation_path + f"information_flow/{model_name}/ChooseCat/val/Image___Question/GQA_val_correct_question_with_categoryThatThisChoose_objThisChoose_ChooseCat_window{window}_Image___Question.csv"
    path2 = operation_path + f"information_flow/{model_name}/ChooseAttr/val/Image___Question/GQA_val_correct_question_with_choose_ChooseAttr_window{window}_Image___Question.csv"
    path3 = operation_path + f"information_flow/{model_name}/ChooseRel/val/Image___Question/GQA_val_correct_question_with_relChooser_ChooseRel_window{window}_Image___Question.csv"
    path4 = operation_path + f"information_flow/{model_name}/CompareAttr/val/Image___Question/GQA_val_correct_question_with_twoCommon_CompareAttr_window{window}_Image___Question.csv"
    path5 = operation_path + f"information_flow/{model_name}/LogicalObj/val/Image___Question/GQA_val_correct_question_with_existThatOr_LogicalObj_window{window}_Image___Question.csv"
    path6 = operation_path + f"information_flow/{model_name}/QueryAttr/val/Image___Question/GQA_val_correct_question_with_positionQuery_QueryAttr_window{window}_Image___Question.csv"

    data1 = read_csv(path=path1)
    data2 = read_csv(path=path2)
    data3 = read_csv(path=path3)
    data4 = read_csv(path=path4)
    data5 = read_csv(path=path5)
    data6 = read_csv(path=path6)

    for ind, (data, path), in enumerate(
            zip([data1, data2, data3, data4, data5, data6], [path1, path2, path3, path4, path5, path6])):
        if ind == 0:
            generate_legend(data,
                            operation_path,
                            model_name=model_name,
                            y="relative diff first",
                            palette=palette[3:4],
                            save_fold="image2question")

        generate_plot(data,
                      save_file=path.replace(".csv", "_first.pdf").replace("GQA_val_correct",
                                                                           f"{model_name}_GQA_val_correct"),
                      operation_path=operation_path,
                      model_name=model_name,
                      y="relative diff first",
                      layer_num=31,
                      palette=palette[3:4],
                      save_fold="image2question")

    # ######################### Image patches to Question #########################
    for ind, task_name in enumerate([task1, task2, task3, task4, task5, task6]):
        data1 = read_csv(
            path=operation_path + f"information_flow/{model_name}/{task_name.split('_')[-1].split('.csv')[0]}/val/Image_Central_Object___Question/{task_name.split('.csv')[0]}_window{window}_Image_Central_Object___Question.csv")
        data2 = read_csv(
            path=operation_path + f"information_flow/{model_name}/{task_name.split('_')[-1].split('.csv')[0]}/val/Image_Without_Central_Object___Question/{task_name.split('.csv')[0]}_window{window}_Image_Without_Central_Object___Question.csv")

        # combine
        data_combined = pd.concat([data1, data2])

        if ind == 0:
            generate_legend(data_combined,
                            operation_path,
                            model_name=model_name,
                            y="relative diff first",
                            palette=palette[4:6],
                            save_fold="imagePatches2question")

        generate_plot(data_combined,
                      save_file=operation_path + f"information_flow/{model_name}/{task_name.split('_')[-1].split('.csv')[0]}/val/{model_name}_{task_name.split('.csv')[0]}_window{window}_Image_patches__Question_firstgen.pdf",
                      operation_path=operation_path,
                      model_name=model_name,
                      y="relative diff first",
                      layer_num=31,
                      palette=palette[4:6],
                      save_fold="imagePatches2question")

        print(f"finish: {task_name}")


# #/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# ####### llava_v1_5_13b #######
model_name="llava_v1_5_13b"
operation_path="output/"
window=9
merge(model_name,operation_path,window)

# ####### llava_v1_5_7b #######
model_name="llava_v1_5_7b"
operation_path="output/"
window=7
merge(model_name,operation_path,window)


# ####### llama3_llava_next_8b #######
model_name="llama3_llava_next_8b"
operation_path="output/"
window=7
merge(model_name,operation_path,window)


# ####### llava_v1_6_vicuna_7b #######
model_name="llava_v1_6_vicuna_7b"
operation_path="output/"
window=7
merge(model_name,operation_path,window)


