"""Fast checks requiring only Python and NumPy, not a model download."""
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from deepmd_demff.manifest import register,load
from prepare_input import prepare


class Inputs(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        weight=self.root/'weights.model';weight.write_bytes(b'fake weights: manifest-only test')
        self.manifest=register(weight,self.root/'weights.demff')
        self.train=self.system('train',0)
        self.valid=self.system('valid',0.1)

    def system(self,name,shift):
        p=self.root/name;d=p/'set.000';d.mkdir(parents=True)
        (p/'type_map.raw').write_text('O H\n');(p/'type.raw').write_text('0 1\n')
        values={'coord':np.array([[1.,1.,1.,2.+shift,1.,1.]]),'box':np.eye(3).reshape(1,9)*7,
                'energy':np.zeros(1),'force':np.zeros((1,6)),'virial':np.zeros((1,9)),'fparam':np.ones((1,1))*.1}
        for k,v in values.items():np.save(d/(k+'.npy'),v)
        return p

    def write(self):
        return prepare(self.manifest,[self.train],[self.valid],self.root/'input.json',steps=12)

    def test_portable_manifest_and_tamper(self):
        dest=self.root/'moved';dest.mkdir()
        for f in ['weights.model','weights.demff']:shutil.copy2(self.root/f,dest/f)
        self.assertEqual(load(dest/'weights.demff')[1],dest/'weights.model')
        (dest/'weights.model').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'SHA256'):load(dest/'weights.demff')

    def test_refuses_overwrite(self):
        with self.assertRaises(FileExistsError):register(self.root/'weights.model',self.manifest)
        self.write()
        with self.assertRaises(FileExistsError):self.write()

    def test_valid_native_input(self):
        d=self.write()
        self.assertEqual(d['model']['type'],'demff')
        self.assertEqual(d['model']['type_map'],['O','H'])
        self.assertEqual(json.loads((self.root/'input.json').read_text())['training']['numb_steps'],12)

    def test_missing_temperature_rejected(self):
        (self.train/'set.000/fparam.npy').unlink()
        with self.assertRaises(FileNotFoundError):self.write()

    def test_negative_temperature_rejected(self):
        np.save(self.train/'set.000/fparam.npy',np.array([[-.1]]))
        with self.assertRaisesRegex(ValueError,'negative fparam'):self.write()

    def test_geometry_overlap_across_dtypes_rejected(self):
        xyz=np.load(self.train/'set.000/coord.npy').astype('float32')
        np.save(self.valid/'set.000/coord.npy',xyz)
        with self.assertRaisesRegex(ValueError,'matching ordered'):self.write()

    def test_type_order_rejected(self):
        (self.valid/'type_map.raw').write_text('H O\n')
        with self.assertRaisesRegex(ValueError,'type_map order'):self.write()

    def test_bad_virial_shape_rejected(self):
        np.save(self.train/'set.000/virial.npy',np.zeros((1,6)))
        with self.assertRaisesRegex(ValueError,'Invalid virial'):self.write()


if __name__=='__main__':unittest.main()
