"""
NSML Server-side module Implementation
"""

import sys
import os
import uuid
import torch


class NSML(type(sys)):

    DATASET_NAME = ''
    DATASET_PATH = ''  # 데이터 경로
    FORCE_CLEANUP = None  # None: 모델 데이터 주피터면 삭제 아니면 놔둠, True: 무조건삭제, False: 무조건놔둠

    HAS_DATASET = True
    IS_ON_NSML = True

    GPU_NUM = torch._C._cuda_getDeviceCount()  # noqa
    user_infer: ...
    user_load: ...
    user_save: ...
    epoch: "[int, type(None)]" = None
    _session: "[str, type(None)]" = None

    @property
    def _force_cleanup(self):
        if self.FORCE_CLEANUP is not None:
            return self.FORCE_CLEANUP
        return 'In' in globals() and 'Out' in globals()

    def _cleanup(self):
        if self._session is not None and self._force_cleanup:
            lookup_path = os.path.join('checkpoints', self._session)
            if os.path.exists(lookup_path):
                import shutil
                shutil.rmtree(lookup_path)
            del self._session

    @property
    def session(self):
        return self._session

    @session.setter
    def session(self, value):
        if self._session is not None and self._force_cleanup:
            self._cleanup()
        self._session = value

    @session.deleter
    def session(self):
        if self._session is not None and self._force_cleanup:
            self._cleanup()
        self._session = None

    def __del__(self):
        if self._session is not None and self._force_cleanup:
            self._cleanup()

    def __init__(self, name):
        super().__init__(name)
        self.user_infer = None
        self.user_save = None
        self.user_load = None
        self.session = None
        self.epoch = 0
        os.makedirs('checkpoints', exist_ok=True)
        # Register jupyter notebook hook
        if self._force_cleanup:
            import atexit
            atexit.register(self._cleanup)

    def report(self, **kwargs):
        if 'epoch' in kwargs:
            self.epoch = kwargs['epoch']

    @staticmethod
    def paused(scope=None): ...

    def save(self, epoch=None, save_fn=None):
        save_fn = save_fn or self.user_save
        iteration = epoch if epoch is not None else self.epoch
        if self.session is None or save_fn is None:
            raise RuntimeError("Unbound runtime")
        if iteration is None:
            raise RuntimeError("Ungiven iteration")
        os.makedirs(os.path.join('checkpoints', self.session), exist_ok=True)
        os.makedirs(os.path.join('checkpoints', self.session, str(iteration)), exist_ok=True)
        save_fn(os.path.join('checkpoints', self.session, str(iteration)))

    def bind(self, save=None, load=None, infer=None):
        if save is not None:
            self.user_save = save
        if load is not None:
            self.user_load = load
        if infer is not None:
            self.user_infer = infer
        self.session = str(uuid.uuid4())
        print("Session Name:", self.session)

    def infer(self, data, **kwargs):
        if not callable(self.user_infer):
            raise RuntimeError("Unbound runtime")
        return self.user_infer(data)

    def load(self, epoch, load_fn=None, session=None):
        load_fn = load_fn or self.user_load
        session = session or self.session
        iteration = epoch
        if session is None or load_fn is None:
            raise RuntimeError("Unbound runtime")
        if iteration is None:
            raise RuntimeError("Ungiven iteration")
        load_fn(os.path.join('checkpoints', self.session, str(iteration)))


sys.modules['nsml'] = NSML('nsml')
