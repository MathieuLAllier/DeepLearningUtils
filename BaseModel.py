import os
import torch
import logging

logger = logging.getLogger(__name__)


class BaseModel:

    """ Parent Class for Saving & Checkpoints Models """

    def __init__(self):
        pass

    def save_model(self, path, post=False):
        """
        :param path: Path Ending with File name & PT extension: example model.pt
        :param post: Whether to return the persistence object instead of saving directly
        """
        with torch.no_grad():

            self.to('cpu')

            # Keeps Modules, Loading Parameters only
            persistence = {k: v for k, v in self.__dict__.items()
                           if (not k.startswith('_') or k == '_modules') and k != 'training'}

            # Extracting Inner Modules necessary
            for modules in persistence['_modules']:
                persistence[modules] = persistence['_modules'][modules]

            del persistence['_modules']
            persistence['state'] = self.state_dict()

            if post:
                return persistence

            try:
                torch.save(persistence, path)
                logger.info('Model saved at {}'.format(path))

            except Exception as e:
                logger.critical('Saving Failed with error message: {{{}}}'.format(e))

    @classmethod
    def load_model(cls, path=None, file=None):
        """
        :param path: Model File path
        :param file: Torch.load output
        :return: Return Model Instance Initiated from saved file
        """

        try:
            persistence = torch.load(path) if not file else file.copy()
            state = persistence.pop('state')
            model = cls(**persistence)
            model.load_state_dict(state)

            logger.info('Loading {} Model from {}'.format(cls.__name__, path if path else 'File'))
            return model

        except Exception as e:
            logger.critical('Loading Model saved with error message: {{{}}}'.format(e))

    def save_checkpoint(self, epoch, train_loss, test_loss, optimizer, path):
        """ AutoEncoder Only """

        try:
            os.makedirs('/'.join(path.split('/')[:-1]))

        except FileExistsError:
            pass

        except Exception as e:
            logger.error('Cannot make Directory: [{}]'.format(e))

        # Save Checkpoint Parameters
        torch.save({
            'model_params': self.save_model('', True),
            'optimizer': optimizer.state_dict(),
            'train_loss': train_loss,
            'test_loss': test_loss,
            'epoch': epoch
        }, path)

        logger.info('Checkpoint {} saved at {}'.format(epoch, path))

    @classmethod
    def load_checkpoint(cls, optimizer, path):
        """
        :param optimizer:
        :param path:
        :return:
        """
        try:
            checkpoint = torch.load(path)

            model = cls.load_model(file=checkpoint.pop('model_params'))
            optimizer.load_state_dict(checkpoint['optimizer'])

            logger.info('Loading Checkpoint at {}'.format(path))

            return model, checkpoint['epoch'], checkpoint['train_loss'], checkpoint['test_loss']

        except Exception as e:
            logger.error('Loading Checkpoint Failed: [{}]'.format(e))