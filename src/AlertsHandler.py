from cfg import *
import time
from multiprocessing import Process, Queue, current_process

import telegram.ext
from telegram.ext.dispatcher import run_async
from telegram.ext import MessageHandler, Filters, CommandHandler, Updater
from telegram import ParseMode
from telegram.error import (TelegramError, Unauthorized, BadRequest,
                            TimedOut, ChatMigrated, NetworkError)


class TelegramAlerts:
        def __init__(self):
            self.process = Process(name='CaptureStream', target=self.connect_and_update, args=())
            self.process.daemon = False
            self.process.start()

        def connect_and_update(self):
            self.updater = Updater(TELEGRAM['token'],
                                   workers=TELEGRAM['workers'],
                                   use_context=True)
            self.bot = telegram.Bot(TELEGRAM['token'])
            self.updater.dispatcher.add_error_handler(self.error_callback)
            start_handler = CommandHandler("start", self.start, pass_args=True, pass_user_data=True)
            self.updater.dispatcher.add_handler(start_handler)

            #self.updater.job_queue.run_repeating(self.check_queue, interval=10, first=0)

            if TELEGRAM['use_polling']:
                LOGGER.info("Telegram: Using long polling.")
                self.updater.start_polling(timeout=15, read_latency=4)
            else:
                LOGGER.info("Telegram: using Webhook")
                self.updater.start_webhook(TELEGRAM['webhook'])

            LOGGER.info("Telegram: Start listening")
            self.check_queue()

        def check_queue(self):
            while True:
                while alert_queue.qsize() > 0:
                    try:
                        alert = alert_queue.get()
                        if alert['type'] == 'video':
                            group_id = TELEGRAM['group_id']
                            vid_name = alert['path'].split('/')[-1]#.replace('_', '\_')
                            LOGGER.info('sending to telegram: {}'.format(vid_name))
                            labels = ', '.join(alert['labels'])
                            self.send_msg(group_id, '📣 ALERT 📣 \n'
                                                    'Date: {} \n'
                                                    'Time: {} \n'
                                                    'Labels: {} \n'
                                                    'Sending video soon...'.format(alert['date'],
                                                                                   alert['time'],
                                                                                   labels))
                            self.bot.send_video(group_id, video=open(alert['path'], 'rb'), supports_streaming=True)
                        if alert['type'] == 'msg':
                            group_id = TELEGRAM['group_id']
                            LOGGER.info('sending to telegram: {}'.format(alert['text']))
                            self.send_msg(group_id, alert['text'])

                    except:
                        try:
                            LOGGER.error('alert_queue_get Error at {}'.format(str(alert)))
                        except:
                            LOGGER.error('alert_queue_get General Error')
                time.sleep(10)



        @run_async
        def start(self, update, context):
            LOGGER.info('-----Start State---- ')
            user_id = str(update.effective_chat.id)
            self.send_msg(user_id, 'Hi There... Nothing else for now', context)


        def split_msg(bot_type, msg):
            max_msg = TELEGRAM['max_msg']
            msg_list = [msg[i:min(i + max_msg, len(msg))] for i in range(0, len(msg), max_msg)]
            return msg_list

        def send_msg(self, user_id, msg, context=None):
            if context == None:
                bot = self.bot
            else:
                bot = context.bot
            msg_list = self.split_msg(msg)
            menu_items = None
            for iii, msg_part in enumerate(msg_list):
                if iii == len(msg_list) - 1:
                    bot.send_message(chat_id=user_id, text=msg_part, reply_markup=menu_items,
                                             parse_mode=ParseMode.MARKDOWN)
                else:
                    bot.send_message(chat_id=user_id, text=msg_part, reply_markup=None,
                                             parse_mode=ParseMode.MARKDOWN)

        def send_error_to_admin(self, update, context):
            errors_text = "ERROR from id: {} --> \n {}"
            msg_out = errors_text.format(str(update.effective_chat.id), str(context.error))
            sender_id = TELEGRAM['admin_id']
            self.send_msg('telegram', sender_id, msg_out, update=update, context=context)

        def error_callback(self, update, context):
            try:
                LOGGER.error('error_callback: context.error ==> ' + str(context.error))
                #send_error_to_admin(update, context)
                raise context.error
            except Unauthorized:
                LOGGER.error('error_callback: Unauthorized')
                pass  # remove update.message.chat_id from conversation list
            except BadRequest:
                LOGGER.error('error_callback: BadRequest')
                pass  # handle malformed requests - read more below!
            except TimedOut:
                LOGGER.error('error_callback: TimedOut')
                pass  # handle slow connection problems
            except NetworkError:
                LOGGER.error('error_callback: NetworkError')
                pass  # handle other connection problems
            except ChatMigrated as e:
                LOGGER.error('error_callback: ChatMigrated')
                pass  # the chat_id of a group has changed, use e.new_chat_id instead
            except TelegramError:
                LOGGER.error('error_callback: TelegramError')
                pass  # handle all other telegram related errors


if __name__ == '__main__':
    # alert_queue.put({
    #     'type': 'video',
    #     'path': '../recordings/final_detection/20201023/114344/20201023_114344_mov_small_2.mkv',
    #     'labels': ['dog', 'person']
    # })
    alert_queue.put({'type': 'video', 'path': '../recordings/final_detection/20201023_162532_mov_small_0.mkv', 'labels': {'person'}})
    tlg = TelegramAlerts()
    while True:
        time.sleep(100000)
