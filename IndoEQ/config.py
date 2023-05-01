from dataclasses import dataclass, field

@dataclass
class TrainingConfig:
    input_hdf5              : str   = ''
    input_csv               : str   = ''
    output_name             : str   = ''
    model_class             : str   = ''
    input_dimension         : tuple = (6000, 3)
    cnn_blocks              : int   = 4
    lstm_blocks             : int   = 3
    padding                 : str   = 'same'
    activation              : str   = 'relu'
    drop_rate               : float = 0.1
    use_prelu               : bool  = False
    shuffle                 : bool  = True
    label_type              : str   = 'triangle'
    normalization_mode      : str   = 'std'
    augmentation            : bool  = False
    add_event_r             : float = 0.6,
    shift_event_r           : float = 0.99,
    add_noise_r             : float = 0.3, 
    drop_channel_r          : float = 0.5,
    add_gap_r               : float = 0.2,
    scale_amplitude_r       : float = None,
    pre_emphasis            : float = False,  
    loss_weights            : list  = field(default_factory=[0.05, 0.40, 0.55])
    loss_types              : list  = field(default_factory=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'])
    train_valid_test_split  : list  = field(default_factory=[0.8, 0.1, 0.1])
    mode                    : str   = 'generator'
    batch_size              : int   = 32
    epochs                  : int   = 20
    monitor                 : str   = 'val_loss'
    patience                : int   = 2
    gpuid                   : int   = None
    key_dim                 : int   = 16
    num_heads               : int   = 8
    reduce_lr_on_plateau    : bool  = False
