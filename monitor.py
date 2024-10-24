import time
import psutil


def monitor_start():
    """
    记录监控开始的时间戳。
    """
    return time.time()


def monitor_end(start_time):
    """
    计算从监控开始到结束所经过的时间。

    参数:
    start_time -- 监控开始的时间戳。
    """
    end_time = time.time()
    print(f"监控耗时：{end_time - start_time}秒")
    return end_time - start_time


def monitor_memory(process=None):
    """
    监控内存使用情况。

    参数:
    process -- 可选的进程对象，默认为当前进程。
    """
    if process is None:
        process = psutil.Process()

    memory_info = process.memory_info()
    return memory_info.rss  # 返回常驻集大小，单位为字节


def monitor_memory_diff(process=None):
    """
    监控内存使用差异。

    参数:
    process -- 可选的进程对象，默认为当前进程。
    """
    if process is None:
        process = psutil.Process()

    before = process.memory_info().rss
    time.sleep(0.1)  # 短暂等待，以便进程有机会使用更多内存
    after = process.memory_info().rss
    return after - before

# 可以添加更多监控函数，例如CPU使用率、GPU使用率等。