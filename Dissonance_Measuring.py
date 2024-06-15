import random
import numpy as np
import sounddevice as sd
import ast
import time
import random as rand
import matplotlib
import matplotlib.pyplot as plot

# 设置默认横坐标系
def x_axis():
    return [2 ** (i * 1/48) for i in range(0, 48)]

# 设置参数
def sine_wave(frequency_spectrum, amplitudes=[1], duration=1, fs=44100):
    duration = duration  # 持续时间（秒）
    fs = fs  # 采样率（Hz）

    amplitudes = [amplitude / sum(amplitudes) / 10 for amplitude in amplitudes]
    # 生成时间序列
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)

    # 生成正弦波信号
    x = np.sum([amp * np.sin(2 * np.pi * freq * t) for freq, amp in zip(frequency_spectrum, amplitudes)], axis=0)

    # 播放信号
    sd.play(x, fs)
    sd.wait()


def test_main(round, interval_1, interval_2):
    # print(interval_1, interval_2)
    random_note_1 = random.randrange(300, 1000) / (interval_1 ** (1 / 2))
    random_note_2 = random_note_1 * interval_1

    random_note_3 = random.randrange(300, 1000) / (interval_2 ** (1 / 2))
    random_note_4 = random_note_3 * interval_2

    if rand.randrange(0,2) == 0:
        sine_wave([random_note_1], duration=1)
        sine_wave([random_note_2], duration=1)
    else:
        sine_wave([random_note_2], duration=1)
        sine_wave([random_note_1], duration=1)

    time.sleep(1)
    if rand.randrange(0,2) == 0:
        sine_wave([random_note_3], duration=1)
        sine_wave([random_note_4], duration=1)
    else:
        sine_wave([random_note_4], duration=1)
        sine_wave([random_note_3], duration=1)

    answer = input("测试次数： " + str(round) + " 哪一个旋律音程更不协和? 前 = 0, 后 = 1: ")
    data_to_be_collected = [answer, random_note_1, random_note_2, random_note_3, random_note_4]
    return data_to_be_collected


def elo(value_A, value_B, A_win_or_lose, difference_scale):
    # right win = 1, left win = 0, 可以为浮点
    # difference_scale = 10
    A_win_or_lose = float(A_win_or_lose)
    value_B = float(value_B)
    value_A = float(value_A)
    if value_A == value_B or A_win_or_lose == 0.5:
        value_A -= difference_scale / 2 * (A_win_or_lose - 0.5) * 2
        value_B += difference_scale / 2 * (A_win_or_lose - 0.5) * 2
        return value_A, value_B
    e = 2.7182818
    difference_between_B_and_A = (value_B - value_A) * -1 * (A_win_or_lose - 0.5) * 2 / difference_scale
    difference_score = e ** difference_between_B_and_A * (
            e ** difference_between_B_and_A - difference_between_B_and_A - 1) / (
                               -1 + e ** difference_between_B_and_A) ** 2 * difference_scale
    value_A -= difference_score * (A_win_or_lose - 0.5) * 2
    value_B += difference_score * (A_win_or_lose - 0.5) * 2
    return value_A, value_B


# print(elo(0, 15, 0.7))

def plot_by_matplotlib(ranking,times,error_bar=20):
    matplotlib.use("TkAgg")
    x_list = x_axis()
    plot.errorbar(x_list, ranking, yerr=error_bar, fmt='-o', ecolor=(0.9, 0.9, 0.9), capsize=5)
    plot.rcParams['font.sans-serif'] = ['DejaVu Sans']
    for i in range(len(x_list)):
        plot.annotate(
            f'({[round(elements, 2) for elements in x_list][i]}, {[round(elements, 0) for elements in ranking][i]})',
            ([round(elements, 2) for elements in x_list][i], [int(round(elements)) for elements in ranking][i]),
            textcoords="offset points", xytext=(0, 10), ha='center', color = "k", fontsize=7)
    plot.rcParams['font.sans-serif'] = ['SimHei']
    plot.rcParams['axes.unicode_minus'] = False
    plot.xticks(np.arange(1, 2.1, 0.1))
    plot.yticks(np.arange(round(min(ranking))-error_bar, round(max(ranking))+5+error_bar, 5))
    plot.xlabel("单个旋律音程中高音与低音频率之比(例如纯五度为1.5)\n"+str([round(elements) for elements in ranking]))
    plot.ylabel("不协和度(高代表不协和)")
    plot.title("您的关于不协和度的测量结果有着±"+str(error_bar)+"的误差(正负1σ)")

    text = "程序开发人员为檬虎Nighten\n有事请找QQ2077030038"
    ax = plot.gca()
    ax.text(0.95, 0.05, text, verticalalignment='bottom', horizontalalignment='right',
            transform=ax.transAxes, fontsize=7, color='black')
    plot.show()
    return

def main():
    print("这是一个关于测试人们对音程感受的程序。让我们开始吧！")
    print("本测试采用了比较法，这是为了规避评分法的高度不确定性，以及尽量减少知识对结论的干扰。请尽量关注于需要测量的概念本身，而非关注于已知的测量答案。")
    print("本实验算法为本人在elo法上改进多次的原创算法，不仅能够比elo算法更快地得到结果，也能显示此刻的误差值。")
    print("由于算法的设计，在约250次之后，会得到较为准确的结果（大约60%的结果会出来），相当于每一个音程大约都被测量了5次。")
    # current_ranking = [random.uniform(0,0.1) for i in range(0,80)]
    # 步长为1
    step = 1
    samples = 48
    step_for_counting_intervals = 1 / 48
    #设置默认elo参数
    difference_scale = 20
    # samples必须为step的整数倍
    # we don't need =1 in detail.
    current_ranking = input("现在的排名是什么，请在此处黏贴（带左右括号不带中文字），如果没有做过本测试输入None: ")
    if current_ranking == "None":
        current_ranking = [0 for i in range(0, samples, step)]
        test_time_list = [0 for i in range(0, samples, step)]
        answer_list = []
        heat = 5
    else:
        test_time_list = ast.literal_eval(input("现在的测试次数是什么，请在此处黏贴（带左右括号不带中文字）"))
        answer_list = ast.literal_eval(input("目前已有的回答是什么，请在此处黏贴（带左右括号不带中文字）"))
        current_ranking = finest_list(answer_list, samples)[0]
        difference_scale = finest_list(answer_list, samples)[1]
        heat = finest_list(answer_list, samples)[4]

    # ensure
    index_and_interval_matching_list = x_axis()

    time_for_testing = len(answer_list)
    # this is starting point, but it is discarded.
    answer = test_main(time_for_testing, index_and_interval_matching_list[random.randrange(0, samples, step)],
                       index_and_interval_matching_list[random.randrange(0, samples, step)])

    while answer[0] != "stop":
        # 考虑次数
        if time_for_testing <= samples*2:
            current_ranking = [items + random.uniform(0, 0.1) for items in current_ranking]

        time_for_testing += 1
        minimum_limitation = round(sum(test_time_list) / len(test_time_list) - 2)
        if min(test_time_list) <= minimum_limitation:
            difference_1 = index_and_interval_matching_list[random.choice(
                [index for index, repeat_time in enumerate(test_time_list) if
                 repeat_time <= minimum_limitation]) * step]
        else:
            if random.randrange(0,2) == 1:
            # sharpness_prepare_list = current_ranking.insert(0, current_ranking[1]).insert(len(current_ranking), current_ranking[len(current_ranking) - 1])
                sharpness_prepare_list = current_ranking[:]
                sharpness_prepare_list.insert(0, current_ranking[1])
                sharpness_prepare_list.insert(len(current_ranking), current_ranking[len(current_ranking) - 1])
                sharpness_list = [0 if abs(sharpness_prepare_list[i + 1] + sharpness_prepare_list[i - 1] - sharpness_prepare_list[i] * 2) <= 15 else abs(sharpness_prepare_list[i + 1] + sharpness_prepare_list[i - 1] - sharpness_prepare_list[i] * 2)
                                  for i in range(1, len(sharpness_prepare_list) - 1)]
                sharpness_list = [abs(elements) for elements in sharpness_list]
                if max(sharpness_list) > 15:
                        difference_1 = index_and_interval_matching_list[sharpness_list.index(random.choice([elements for elements in sharpness_list if elements > 15]))]
                else:
                    difference_1 = index_and_interval_matching_list[random.randrange(0, samples, step)]
            else:
                difference_1 = index_and_interval_matching_list[random.randrange(0, samples, step)]
        minimum_list = [
            1000 if index_and_interval_matching_list[current_ranking.index(items) * step] == difference_1 else abs(
                items - current_ranking[index_and_interval_matching_list.index(difference_1) * step]) for items in
            current_ranking]
        # print(minimum_list)
        # print( difference_1, minumum_list.index(min(minumum_list)) * step)

        # getting answer and make it into a list
        difference_2 = index_and_interval_matching_list[minimum_list.index(min(minimum_list))]
        answer = test_main(time_for_testing, difference_1, difference_2)
        answer_list.append(answer)

        if answer[0] == "stop":
            # 只需要复制current_ranking。
            print("\n以下内容请妥善保存：")
            print("当前排名", current_ranking)
            print("测试次数", test_time_list)
            print("回答列表" ,answer_list[0:len(answer_list) - 1])
            print("请黏贴并妥善保存，以便随后继续用")
            plot_by_matplotlib(current_ranking,time_for_testing, error_bar= difference_scale * 1.81)
            # print(test_time_list)
            return
        # print(current_ranking)
        # lost_win_list = [[int(sublists[0]), sublists[2] - sublists[1], sublists[4] - sublists[3]] for sublists in answer_list]
        test_time_list[index_and_interval_matching_list.index(difference_1)] += 1
        test_time_list[minimum_list.index(min(minimum_list))] += 1

        # 使用answer，此处需要对不同interval做出不同调整
        answer = [int(items) if answer.index(items) != 0 else float(items) for items in answer]
        # 知道是第几位的elo
        elo_final_list = elo(current_ranking[index_and_interval_matching_list.index(difference_1)],
                             current_ranking[index_and_interval_matching_list.index(difference_2)], answer[0], difference_scale)
        current_ranking[index_and_interval_matching_list.index(difference_1)] = elo_final_list[0]
        current_ranking[index_and_interval_matching_list.index(difference_2)] = elo_final_list[1]

        difference_scale = finest_list(answer_list, samples)[1]

#对未知样本进行elo_list的处理:，得到一个全新的list
def elo_final_list_when_unknown(answer_list, number_of_items_in_the_list, heat):
    step = 1
    step_for_counting_intervals = 1/48
    samples = number_of_items_in_the_list
    difference_scale = 10
    current_ranking = [0 for i in range(0, samples, step)]
    index_and_interval_matching_list = x_axis()

    for i in range(0, len(answer_list)):
        difference_1 = answer_list[i][2]/answer_list[i][1]
        difference_2 = answer_list[i][4]/answer_list[i][3]
        answer = answer_list[i]
        elo_final_list = elo(current_ranking[np.where(np.isclose(index_and_interval_matching_list, difference_1))[0][0]],
                             current_ranking[np.where(np.isclose(index_and_interval_matching_list, difference_2))[0][0]], answer[0], difference_scale)
        current_ranking[np.where(np.isclose(index_and_interval_matching_list, difference_1))[0][0]] = elo_final_list[0]
        current_ranking[np.where(np.isclose(index_and_interval_matching_list, difference_2))[0][0]] = elo_final_list[1]

        if i == samples:
            past_elo_list = current_ranking.copy()
        time_gallop = [round(samples * 1.2 ** i) for i in range(0, 30)]
        if i > samples and i in time_gallop:
            difference_scale = difference_scale *(1/ np.std(current_ranking) * np.std(past_elo_list)) ** heat
            past_elo_list = current_ranking.copy()
    return [current_ranking, difference_scale, difference_scale/np.std(current_ranking), difference_scale * 1.81, heat]

def finest_list(answer_list, number_of_items_in_the_list):
    better_list = []
    for i in range(1,15):
        better_list.append(elo_final_list_when_unknown(answer_list, number_of_items_in_the_list, i)[2])
    heat_needed = better_list.index(min(better_list))
    return elo_final_list_when_unknown(answer_list, number_of_items_in_the_list, heat_needed)


main()
input("请在保存后手动关闭程序")
