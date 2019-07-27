#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import configparser


class Config():

    config_filepath = './'

    learning_rate = 0.001
    batch_size = 32
    epochs = 10


current_config = Config()

cf = configparser.ConfigParser()
cf.read(current_config.config_filepath)
sections = cf.sections()

for k,section in enumerate(sections):
    user_config = cf.items(section)
    for k2,v in enumerate(user_config):
        current_config.__setattr__(v[0], v[1])
