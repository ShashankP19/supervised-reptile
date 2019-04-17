"""
Train a model on miniImageNet.
"""

import random

import tensorflow as tf

from supervised_reptile.args import argument_parser, model_kwargs, train_kwargs, evaluate_kwargs
from supervised_reptile.eval import evaluate
from supervised_reptile.models import MiniImageNetModel, RegularizedMiniImageNetModel
from supervised_reptile.miniimagenet import read_dataset
from supervised_reptile.train import train

DATA_DIR = 'data/miniimagenet'

def run_miniimagenet(model, args, train_set, val_set, test_set, checkpoint, title):
    
    print("\nTraining phase of " + title + "\n")
    
    with tf.Session() as sess:
        if not args.pretrained:
            print('Training...')
            train(sess, model, train_set, test_set, checkpoint, **train_kwargs(args))
        else:
            print('Restoring from checkpoint...')
            tf.train.Saver().restore(sess, tf.train.latest_checkpoint(checkpoint))

        print("\nEvaluation phase of " + title + "\n")
        print('Evaluating...')
        eval_kwargs = evaluate_kwargs(args)
        print('Train accuracy: ' + str(evaluate(sess, model, train_set, **eval_kwargs)))
        print('Validation accuracy: ' + str(evaluate(sess, model, val_set, **eval_kwargs)))
        print('Test accuracy: ' + str(evaluate(sess, model, test_set, **eval_kwargs)))


def main():
    """
    Load data and train a model on it.
    """
    args = argument_parser().parse_args()
    random.seed(args.seed)

    train_set, val_set, test_set = read_dataset(DATA_DIR)

    if args.org:
        model = MiniImageNetModel(args.classes, **model_kwargs(args))
        run_miniimagenet(model, args, train_set, val_set, test_set, args.checkpoint_org, "MODEL WITHOUT REGULARIZATION")
            
    tf.reset_default_graph()

    if args.reg:
        reg_model = RegularizedMiniImageNetModel(args.classes, **model_kwargs(args))
        run_miniimagenet(reg_model, args, train_set, val_set, test_set, args.checkpoint_reg, "MODEL WITH REGULARIZATION")
       
        

if __name__ == '__main__':
    main()
