

baseDir = '/cifs/diedrichsen/data/Chord_exp/EFC_learningTMS'
behavDir = 'behavioural'
pcmDir  = 'pcm'
rsaDir = 'RSA'
sampling_rate = 100 # sampling rate in Hz

channels = ['finger_1_force_x',	'finger_1_force_y',	'finger_1_force_z',
'finger_2_force_x',	'finger_2_force_y',	'finger_2_force_z',
'finger_3_force_x',	'finger_3_force_y',	'finger_3_force_z',
'finger_4_force_x',	'finger_4_force_y',	'finger_4_force_z',
'finger_5_force_x',	'finger_5_force_y',	'finger_5_force_z',	]

channels_PRE = ['finger_1_force_x_PRE',	'finger_1_force_y_PRE',	'finger_1_force_z_PRE',
'finger_2_force_x_PRE',	'finger_2_force_y_PRE',	'finger_2_force_z_PRE',
'finger_3_force_x_PRE',	'finger_3_force_y_PRE',	'finger_3_force_z_PRE',
'finger_4_force_x_PRE',	'finger_4_force_y_PRE',	'finger_4_force_z_PRE',
'finger_5_force_x_PRE',	'finger_5_force_y_PRE',	'finger_5_force_z_PRE',	]

chord_mapping = {
    'C1': 'C1',
    'C2': 'C1r',
    'C3': 'C2',
    'C4': 'C2r',
}

channel_mapping = {
    'finger_1_force_x': 'thumbX',	'finger_1_force_y': 'thumbY',	'finger_1_force_z': 'thumbZ',
'finger_2_force_x': 'indexX',	'finger_2_force_y': 'indexY',	'finger_2_force_z': 'indexZ',
'finger_3_force_x': 'middleX',	'finger_3_force_y': 'middleY',	'finger_3_force_z': 'middleZ',
'finger_4_force_x': 'ringX',	'finger_4_force_y': 'ringY',	'finger_4_force_z': 'ringZ',
'finger_5_force_x': 'pinkieX',	'finger_5_force_y': 'pinkieY',	'finger_5_force_z': 'pinkieZ',
}

channel_mapping_PRE = {
    'finger_1_force_x_PRE': 'thumbX',	'finger_1_force_y_PRE': 'thumbY',	'finger_1_force_z_PRE': 'thumbZ',
'finger_2_force_x_PRE': 'indexX',	'finger_2_force_y_PRE': 'indexY',	'finger_2_force_z_PRE': 'indexZ',
'finger_3_force_x_PRE': 'middleX',	'finger_3_force_y_PRE': 'middleY',	'finger_3_force_z_PRE': 'middleZ',
'finger_4_force_x_PRE': 'ringX',	'finger_4_force_y_PRE': 'ringY',	'finger_4_force_z_PRE': 'ringZ',
'finger_5_force_x_PRE': 'pinkieX',	'finger_5_force_y_PRE': 'pinkieY',	'finger_5_force_z_PRE': 'pinkieZ',
}