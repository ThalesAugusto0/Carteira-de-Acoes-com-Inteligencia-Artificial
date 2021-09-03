#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This program is dedicated to the public domain under the CC0 license.

"""
Bot simples para responder às mensagens do Telegram.

Primeiro, algumas funções do manipulador são definidas. Então, essas funções são passadas para
pelo Despachante e registrados em seus respectivos locais.
Em seguida, o bot é iniciado e executado até pressionarmos Ctrl-C na linha de comando.

Uso:
Exemplo básico do Echobot, repete mensagens.
Pressione Ctrl-C na linha de comando ou envie um sinal ao processo para interromper o
robô.
"""

import logging

from telegram.ext import Updater, CommandHandler, MessageHandler, Filters

# Habilitar registro
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)


# Defina alguns manipuladores de comando. Estes geralmente levam os dois argumentos update e
# contexto. Os manipuladores de erro também recebem o objeto TelegramError gerado com erro.
def start(update, context):
    """Envie uma mensagem quando o comando / start for emitido."""
    update.message.reply_text('Bem vindo!')


def help(update, context):
    """Envie uma mensagem quando o comando / ajuda for emitido."""
    update.message.reply_text('Help!')


def echo(update, context):
    """Echo the user message."""
    update.message.reply_text(update.message.text)


def error(update, context):
    """Erros de log causados por atualizações."""
    logger.warning('Update "%s" caused error "%s"', update, context.error)


def main():
    """Inicie o bot."""
    # Crie o Updater e passe a ele o token do seu bot.
    # Certifique-se de definir use_context = True para usar os novos callbacks baseados em contexto
    # Poste a versão 12, isso não será mais necessário
    updater = Updater("1919779488:AAFQEYvus8E3W2LuV_KRZ0NG5W1X_051mkU", use_context=True)

    # Faça com que o despachante registre manipuladores
    dp = updater.dispatcher

    # em comandos diferentes - responder no telegrama
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("help", help))

    # no noncommand, ou seja, mensagem - ecoa a mensagem no Telegram
    dp.add_handler(MessageHandler(Filters.text, echo))

    # registrar todos os erros
    dp.add_error_handler(error)

    # Inicie o bot
    updater.start_polling()

    # Execute o bot até pressionar Ctrl-C ou o processo recebe SIGINT,
    # SIGTERM ou SIGABRT. Isso deve ser usado na maioria das vezes, uma vez que
    # start_polling () não bloqueia e irá parar o bot normalmente.
    updater.idle()


if __name__ == '__main__':
    main()