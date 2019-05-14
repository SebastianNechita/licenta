from time import time
from typing import Sequence

import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, LearningRateScheduler

from yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss
from yolo3.utils import get_random_data

from pedect.trainer.Trainer import Trainer
from pedect.utils.constants import *

class LRTensorBoard(TensorBoard):
    def __init__(self, log_dir):  # add other arguments to __init__ if you need
        super().__init__(log_dir=log_dir)

    def on_epoch_end(self, epoch, logs=None):
        logs.update({'lr': K.eval(self.model.optimizer.lr)})
        super().on_epoch_end(epoch, logs)


class YoloTrainer(Trainer):
    def __init__(self, config, annotationFiles: Sequence[str]=None):
        if annotationFiles is None:
            annotationFiles = [ANNOTATIONS_FILE]
        self.config = config
        self.annotationFiles = annotationFiles

    def createAnnotationFile(self) -> str:
        path = os.path.join(BASE_DIR, 'temporary_annotations_file.csv')
        writeFile = open(path, "w+")
        for aFile in self.annotationFiles:
            f = open(aFile, 'r')
            for line in f.readlines():
                if len(line) >= 2:
                    writeFile.write(line)
            f.close()
        writeFile.close()
        return path

    def train(self) -> None:
        config = self.config
        print(config)
        freezeNoEpochs = config.freezeNoEpochs if config.loadPreTrained else 0
        noFreezeNoEpochs = config.noFreezeNoEpochs
        is_tiny_version = config.isTiny  # default setting

        annotation_path = self.createAnnotationFile()

        # annotation_path = ANNOTATIONS_FILE
        models_dir = os.path.join(MODELS_DIR, config.trainId)
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        config.save()
        config.saveText()
        classes_path = LABELS_FILE
        anchors_path = config.getAnchorsPath()
        class_names = get_classes(classes_path)
        num_classes = len(class_names)
        anchors = get_anchors(anchors_path)

        input_shape = config.inputShape
        tensorboard = LRTensorBoard(log_dir=os.path.join(MODELS_DIR, str(config.trainId), "logs/{}".format(config.trainId)))
        preTrainedModelPath = config.preTrainedModelPath
        if preTrainedModelPath == "default":
            preTrainedModelPath = 'tiny_yolo_weights.h5' if is_tiny_version else 'yolo_weights.h5'
            preTrainedModelPath = os.path.join(YOLO_DIR, 'model_data', preTrainedModelPath)
        createTheModel = create_tiny_model if is_tiny_version else create_model
        model = createTheModel(input_shape, anchors, num_classes, load_pretrained=config.loadPreTrained, freeze_body=2,
                               weights_path=preTrainedModelPath)
        # if is_tiny_version:
        #     model = create_tiny_model(input_shape, anchors, num_classes,
        #                               load_pretrained=config.loadPretrained,
        #                               freeze_body=2,
        #                               weights_path=os.path.join(YOLO_DIR, 'model_data', pretrainedModelName))
        # else:
        #     model = create_model(input_shape, anchors, num_classes,
        #                          load_pretrained=config.loadPretrained,
        #                          freeze_body=2, weights_path=os.path.join(YOLO_DIR, 'model_data',
        #                                                                   pretrainedModelName))
        # make sure you know what you freeze

        # logging = TensorBoard(log_dir=models_dir)
        checkpoint = ModelCheckpoint(os.path.join(models_dir, 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'),
                                     monitor='val_loss', save_weights_only=True, save_best_only=True,
                                     period=config.checkpointPeriod)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

        val_split = config.validationSplit
        with open(annotation_path) as f:
            lines = f.readlines()
        np.random.seed(10101)
        np.random.shuffle(lines)
        np.random.seed(None)
        num_val = int(len(lines) * val_split)
        num_train = len(lines) - num_val
        # Train with frozen layers first, to get a stable loss.
        # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
        if freezeNoEpochs > 0:
            print("In freeze phase: FreezeNoEpochs = %d" % freezeNoEpochs)
            model.compile(optimizer=Adam(lr=1e-3), loss={
                # use custom yolo_loss Lambda layer.
                'yolo_loss': lambda y_true, y_pred: y_pred})

            batch_size = config.freezeBatchSize
            print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val,
                                                                                       batch_size))
            
            model.fit_generator(
                data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
                steps_per_epoch=max(1, num_train // batch_size),
                validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors,
                                                       num_classes),
                validation_steps=max(1, num_val // batch_size),
                epochs=freezeNoEpochs + config.alreadyTrainedEpochs,
                initial_epoch=config.alreadyTrainedEpochs,
                callbacks=[checkpoint, tensorboard])
        #         model.save_weights(log_dir + 'trained_weights_stage_1.h5')

        # Unfreeze and continue training, to fine-tune.
        # Train longer if the result is not good.
        if noFreezeNoEpochs > 0:
            print("In no freeze phase: NoFreezeNoEpochs = %d" % noFreezeNoEpochs)

            for i in range(len(model.layers)):
                model.layers[i].trainable = True
            model.compile(optimizer=Adam(lr=config.initialLR),
                          loss={'yolo_loss': lambda y_true, y_pred: y_pred})  # recompile to apply the change
            print('Unfreeze all of the layers.')

            batch_size = config.noFreezeBatchSize  # note that more GPU memory is required after unfreezing the body
            print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val,
                                                                                       batch_size))

            model.fit_generator(
                data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
                steps_per_epoch=max(1, num_train // batch_size),
                validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors,
                                                       num_classes),
                validation_steps=max(1, num_val // batch_size),
                epochs=freezeNoEpochs + noFreezeNoEpochs + config.alreadyTrainedEpochs,
                initial_epoch=freezeNoEpochs + config.alreadyTrainedEpochs,
                callbacks=[checkpoint, tensorboard, reduce_lr, early_stopping])
        model.save_weights(config.getModelPath())
        config.alreadyTrainedEpochs += freezeNoEpochs + noFreezeNoEpochs
        config.preTrainedModelPath = config.getModelPath()
        config.loadPreTrained = True
        config.save()
        config.saveText()
        # Further training if needed.
        print("Finished training!")


def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)

def create_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
                 weights_path='model_data/yolo_weights.h5'):
    '''create the training model'''
    K.clear_session()  # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h // {0: 32, 1: 16, 2: 8}[l], w // {0: 32, 1: 16, 2: 8}[l],
                           num_anchors // 3, num_classes + 5)) for l in range(3)]

    model_body = yolo_body(image_input, num_anchors // 3, num_classes)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze darknet53 body or freeze all but 3 output layers.
            num = (185, len(model_body.layers) - 3)[freeze_body - 1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model

def create_tiny_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
                      weights_path='model_data/tiny_yolo_weights.h5'):
    '''create the training model, for Tiny YOLOv3'''
    K.clear_session()  # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h // {0: 32, 1: 16}[l], w // {0: 32, 1: 16}[l],
                           num_anchors // 2, num_classes + 5)) for l in range(2)]

    model_body = tiny_yolo_body(image_input, num_anchors // 2, num_classes)
    print('Create Tiny YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze the darknet body or freeze all but 2 output layers.
            num = (20, len(model_body.layers) - 2)[freeze_body - 1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.7})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model

def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i == 0:
                np.random.shuffle(annotation_lines)
            image, box = get_random_data(annotation_lines[i], input_shape, random=True)
            image_data.append(image)
            box_data.append(box)
            i = (i + 1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)

def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes):
    n = len(annotation_lines)
    if n == 0 or batch_size <= 0: return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)
