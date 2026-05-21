#!/usr/bin/env python3
"""
生成 100 条中文语音测试音频（使用 macOS 内置 TTS）
"""
import os
import subprocess
import sys

# 100 条中文测试语句（覆盖数字、日常对话、命令、绕口令等）
TEST_CASES = [
    # ===== 数字（1-10）=====
    ("零一二三四五六七八九十", "numbers_01"),
    ("一百二十三", "numbers_02"),
    ("三千五百六十七", "numbers_03"),
    ("九万八千七百六十五", "numbers_04"),
    ("一二三四五六七八九零", "numbers_05"),
    ("百分之十五", "numbers_06"),
    ("三点一四一五九二六", "numbers_07"),
    ("二零二六年五月二十一日", "numbers_08"),
    ("我的手机号是一三八零零零一二三四五", "numbers_09"),
    ("第九百九十九号", "numbers_10"),

    # ===== 问候/礼貌（11-20）=====
    ("你好", "greet_01"),
    ("大家好", "greet_02"),
    ("早上好", "greet_03"),
    ("下午好", "greet_04"),
    ("晚上好", "greet_05"),
    ("谢谢", "greet_06"),
    ("非常感谢", "greet_07"),
    ("不客气", "greet_08"),
    ("对不起", "greet_09"),
    ("没关系", "greet_10"),

    # ===== 命令/操作（21-30）=====
    ("请打开窗户", "cmd_01"),
    ("请关闭灯光", "cmd_02"),
    ("请播放音乐", "cmd_03"),
    ("请暂停播放", "cmd_04"),
    ("请提高音量", "cmd_05"),
    ("请降低温度", "cmd_06"),
    ("帮我叫一辆出租车", "cmd_07"),
    ("请发送这条消息", "cmd_08"),
    ("帮我删除这个文件", "cmd_09"),
    ("请保存当前文档", "cmd_10"),

    # ===== 日常短语（31-40）=====
    ("今天天气真好", "daily_01"),
    ("明天会下雨吗", "daily_02"),
    ("请问现在几点了", "daily_03"),
    ("我要去火车站", "daily_04"),
    ("这顿饭很好吃", "daily_05"),
    ("最近工作很忙", "daily_06"),
    ("周末一起吃饭吧", "daily_07"),
    ("好久不见", "daily_08"),
    ("祝你生日快乐", "daily_09"),
    ("一路平安", "daily_10"),

    # ===== 人名（41-50）=====
    ("我叫张三", "name_01"),
    ("她是李四", "name_02"),
    ("王五是我的朋友", "name_03"),
    ("赵六在吗", "name_04"),
    ("请问陈小姐在吗", "name_05"),
    ("刘老师您好", "name_06"),
    ("小明的成绩很好", "name_07"),
    ("李华和王芳结婚了", "name_08"),
    ("周杰伦的新歌很好听", "name_09"),
    ("马化腾是腾讯的创始人", "name_10"),

    # ===== 科技/指令（51-60）=====
    ("打开浏览器", "tech_01"),
    ("搜索天气预报", "tech_02"),
    ("查看我的邮件", "tech_03"),
    ("设置闹钟七点半", "tech_04"),
    ("导航到最近的加油站", "tech_05"),
    ("连接无线网络", "tech_06"),
    ("蓝牙已打开", "tech_07"),
    ("备份我的照片", "tech_08"),
    ("下载这个应用程序", "tech_09"),
    ("更新操作系统", "tech_10"),

    # ===== 长句（61-70）=====
    ("请不吝点赞订阅转发打赏支持明镜与点点栏目", "long_01"),
    ("今天下午三点在会议室开会", "long_02"),
    ("我想预订一张明天飞往北京的机票", "long_03"),
    ("请帮我查一下从上海到深圳的高铁时刻表", "long_04"),
    ("这个月的销售额比上个月增长了百分之二十", "long_05"),
    ("人工智能正在改变我们的生活和工作方式", "long_06"),
    ("请把这份文件打印三份并装订好", "long_07"),
    ("明天的天气是多云转晴温度在十五到二十度之间", "long_08"),
    ("我正在学习如何使用深度学习框架进行语音识别", "long_09"),
    ("这家餐厅的招牌菜是红烧肉和清蒸鲈鱼", "long_10"),

    # ===== 同音词（71-80）=====
    ("树木和数目", "homo_01"),
    ("公式和攻势", "homo_02"),
    ("权利和权力", "homo_03"),
    ("实事和实施", "homo_04"),
    ("必须和必需", "homo_05"),
    ("长年和常年", "homo_06"),
    ("会面和汇面", "homo_07"),
    ("近期和禁区", "homo_08"),
    ("理解和解离", "homo_09"),
    ("机密和机密的区别", "homo_10"),

    # ===== 吉祥话（81-90）=====
    ("好事成双", "bless_01"),
    ("心想事成", "bless_02"),
    ("万事如意", "bless_03"),
    ("一帆风顺", "bless_04"),
    ("大吉大利", "bless_05"),
    ("恭喜发财", "bless_06"),
    ("身体健康", "bless_07"),
    ("学业进步", "bless_08"),
    ("工作顺利", "bless_09"),
    ("阖家幸福", "bless_10"),

    # ===== 绕口令/难句（91-100）=====
    ("吃葡萄不吐葡萄皮不吃葡萄倒吐葡萄皮", "tongue_01"),
    ("四是四十是十十四是十四四十是四十", "tongue_02"),
    ("黑化肥挥发发灰会花飞灰化肥挥发发黑会飞花", "tongue_03"),
    ("红鲤鱼与绿鲤鱼与驴", "tongue_04"),
    ("妈妈赶马马慢妈妈骂马", "tongue_05"),
    ("妞妞骑牛牛拧妞妞拧牛", "tongue_06"),
    ("山前有四十四个小狮子", "tongue_07"),
    ("班干管班干班干帮班干", "tongue_08"),
    ("老龙恼怒闹老农老农恼怒闹老龙", "tongue_09"),
    ("牛郎年年念刘娘刘娘年年念牛郎", "tongue_10"),
]

def generate_audio(text, output_path, voice="Tingting"):
    """使用 macOS say 命令生成中文语音并转为 16kHz WAV"""
    aiff_path = output_path + ".aiff"
    # 生成 AIFF
    subprocess.run(
        ["say", "-v", voice, text, "-o", aiff_path],
        check=True, capture_output=True
    )
    # 转为 16kHz 16-bit mono WAV
    subprocess.run(
        ["afconvert", "-f", "WAVE", "-d", "LEI16", "-r", "16000", aiff_path, output_path],
        check=True, capture_output=True
    )
    # 删除临时 AIFF
    os.remove(aiff_path)

def main():
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "test_audio"
    voice = sys.argv[2] if len(sys.argv) > 2 else "Tingting"

    os.makedirs(output_dir, exist_ok=True)

    # 写入 ground truth 文件
    gt_path = os.path.join(output_dir, "ground_truth.txt")
    with open(gt_path, "w", encoding="utf-8") as f:
        for text, case_id in TEST_CASES:
            f.write(f"{case_id}|{text}\n")

    total = len(TEST_CASES)
    for i, (text, case_id) in enumerate(TEST_CASES, 1):
        wav_path = os.path.join(output_dir, f"{case_id}.wav")
        if os.path.exists(wav_path):
            print(f"[{i}/{total}] SKIP (exists): {case_id} - {text}")
            continue
        print(f"[{i}/{total}] Generating: {case_id} - {text}")
        try:
            generate_audio(text, wav_path, voice)
            # 获取文件大小验证
            size = os.path.getsize(wav_path)
            print(f"  -> OK ({size} bytes)")
        except Exception as e:
            print(f"  -> FAILED: {e}")

    print(f"\nDone! {total} audio files generated in '{output_dir}'")
    print(f"Ground truth saved to '{gt_path}'")

if __name__ == "__main__":
    main()