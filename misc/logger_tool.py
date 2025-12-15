import sys
import time

""""
这段代码定义了两个类，Logger 和 Timer，它们用于日志记录和时间跟踪。

Logger 类
Logger 类用于同时将信息打印到控制台和写入到一个日志文件中。它具有以下方法：

__init__(self, outfile): 构造函数，接收一个日志文件的路径 outfile。它还会记录日志文件创建的时间。
write(self, message): 将传入的消息 message 打印到控制台，并追加到日志文件中。
write_dict(self, dict): 接收一个字典，将字典中的键值对格式化为字符串，并调用 write 方法将其写入日志。
write_dict_str(self, dict): 类似于 write_dict，但用于记录字符串类型的值。
flush(self): 清空输出，确保所有内容都已写入日志文件。
Timer 类
Timer 类用于跟踪时间，提供时间估计功能。它具有以下方法和属性：

__init__(self, starting_msg=None): 构造函数，可以接收一个起始消息 starting_msg，如果提供，将打印该消息和当前时间。
start: 类属性，记录总时间的开始。
stage_start: 类属性，记录当前阶段的开始。
elapsed: 已过去的时间。
est_total: 估计的总时间。
est_remaining: 估计的剩余时间。
est_finish: 估计的完成时间（以时间戳形式）。
__enter__(self) 和 __exit__(self, exc_type, exc_val, exc_tb): 使 Timer 可以作为上下文管理器使用。
update_progress(self, progress): 更新进度，并重新计算估计的总时间、剩余时间和预计完成时间。
str_estimated_complete(self): 返回估计完成时间的字符串表示。
str_estimated_remaining(self): 返回估计剩余时间的字符串表示，单位为小时。
estimated_remaining(self): 返回估计剩余时间（小时）。
get_stage_elapsed(self): 获取当前阶段已过去的时间。
reset_stage(self): 重置当前阶段的开始时间。
lapse(self): 计算自当前阶段开始以来经过的时间，并重置阶段开始时间。
这些类可以在需要记录操作日志和跟踪操作时间的应用程序中使用。例如，在训练机器学习模型时，可以使用 Logger 类记录训练过程中的参数和性能指标，使用 Timer 类跟踪训练阶段的时间。



"""
# Logger是一个用于日志记录的自定义类，封装了日志消息的写入操作，同时将消息输出到控制台和文件。
class Logger(object):
    def __init__(self, outfile):
        # outfile 指定日志文件的路径
        # self.terminal存储标准输出（通常是控制台）的引用，以便于日志可以同时打印到控制台
        self.terminal = sys.stdout
        # self.log_path存储日志的路径
        self.log_path = outfile
        # 获得当前时间并格式化为2字符串
        now = time.strftime("%c")
        # 第一个self.write调用会将当前时间写入日志文件，并标记日志的开始
        self.write('================ (%s) ================\n' % now)
    # write方法将传入的消息message写到终端和日志文件
    def write(self, message):
        self.terminal.write(message)
        with open(self.log_path, mode='a', encoding='utf-8') as f:
            f.write(message)
    # 接受一个字典，将字典每一项格式化为字符串  然后用self.write方法将格式化后的字符串写入日志
    def write_dict(self, dict):
        message = ''
        for k, v in dict.items():
            message += '%s: %.7f ' % (k, v)
        self.write(message)
    # 与 write_dict 类似，但这里假定字典的值是字符串类型，将字典的每一项格式化为字符串，键和值之间用冒号和空格分隔。
    def write_dict_str(self, dict):
        message = ''
        for k, v in dict.items():
            message += '%s: %s ' % (k, v)
        self.write(message)
    # flush 方法：清空缓存，确保所有日志消息都被写入到日志文件中。这在需要确保日志立即写入磁盘时很有用
    def flush(self):
        self.terminal.flush()

# 这个 Timer 类是一个用于跟踪和估计任务完成时间的自定义工具类
class Timer:
    # __init__ 方法：类的构造函数，初始化 Timer 实例。
    # starting_msg：可选参数，用于在实例化时打印开始消息。
    # self.start：记录 Timer 实例创建时的时间戳。
    # self.stage_start：记录当前阶段的开始时间戳。
    # 如果提供了 starting_msg，则会打印这个消息和当前时间。
    def __init__(self, starting_msg = None):
        self.start = time.time()
        self.stage_start = self.start

        if starting_msg is not None:
            print(starting_msg, time.ctime(time.time()))
    # __enter__ 方法：使得 Timer 可以用于上下文管理器（with 语句）。返回 Timer 实例本身。
    def __enter__(self):
        return self
    # __exit__ 方法：上下文管理器的退出方法，当前实现为空，用于可能的异常处理。
    def __exit__(self, exc_type, exc_val, exc_tb):
        return
    # update_progress 方法：更新进度并估计剩余时间。
    # progress：表示任务完成的进度比例。
    # self.elapsed：已过去的时间。
    # self.est_total：估计的总时间。
    # self.est_remaining：估计的剩余时间。
    # self.est_finish：估计的完成时间，以时间戳表示。
    def update_progress(self, progress):
        self.elapsed = time.time() - self.start
        self.est_total = self.elapsed / progress
        self.est_remaining = self.est_total - self.elapsed
        self.est_finish = int(self.start + self.est_total)

    # str_estimated_complete 方法：返回估计的完成时间的字符串表示，格式化为可读的时间格式。
    def str_estimated_complete(self):
        return str(time.ctime(self.est_finish))
    # str_estimated_remaining 方法：返回估计的剩余时间的字符串表示，以小时为单位。
    def str_estimated_remaining(self):
        return str(self.est_remaining/3600) + 'h'
    # estimated_remaining 方法：返回估计的剩余时间，以小时为单位。
    def estimated_remaining(self):
        return self.est_remaining/3600
    # get_stage_elapsed 方法：获取当前阶段已过去的时间，以秒为单位。
    def get_stage_elapsed(self):
        return time.time() - self.stage_start
    # reset_stage 方法：重置当前阶段的开始时间。
    def reset_stage(self):
        self.stage_start = time.time()
    # lapse 方法：获取自当前阶段开始以来经过的时间，并重置阶段开始时间。
    def lapse(self):
        out = time.time() - self.stage_start
        self.stage_start = time.time()
        return out
