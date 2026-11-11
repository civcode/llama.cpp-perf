import sqlite3
import tempfile
import unittest
from pathlib import Path
from rocpd_operator_compare import read_trace, kernel_family

class TestTrace(unittest.TestCase):
    def test_classify(self):
        self.assertEqual(kernel_family('void lightning_indexer_kernel_vec<float>'), 'qsa_lightning_indexer')
        self.assertEqual(kernel_family('void top_k_mask_kernel'), 'qsa_top_k_mask')
        self.assertEqual(kernel_family('void mul_mat_vec_q<...>'), 'quantized_matmul')
        self.assertEqual(kernel_family('void gated_delta_net_cuda<...>'), 'recurrent_gdn_ssm')

    def test_trace(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'trace.db'
            db=sqlite3.connect(path)
            db.executescript('''
              CREATE TABLE rocpd_info_agent (id INT, name TEXT, type TEXT, absolute_index INT);
              INSERT INTO rocpd_info_agent VALUES (0,'Ryzen','CPU',0),(1,'gfx1201','GPU',1);
              CREATE TABLE kernels (agent_abs_index INT,start INT,end INT,duration INT,name TEXT);
              INSERT INTO kernels VALUES (1,0,10,10,'start marker');
              CREATE TABLE rocpd_memory_copy (start INT,end INT,size INT,src_agent_id INT,dst_agent_id INT);
              CREATE TABLE regions_and_samples (start INT,end INT,name TEXT,category TEXT);
              INSERT INTO regions_and_samples VALUES (400,500,'hipStreamSynchronize','HIP_RUNTIME_API_EXT');
            ''')
            for i in range(8):
                pos=1000000000+i*100000000
                db.execute('INSERT INTO rocpd_memory_copy VALUES (?,?,?,?,?)',(pos,pos+20000,993280,1,0))
                db.execute('INSERT INTO kernels VALUES (?,?,?,?,?)', (1,pos+1000,pos+11000,10000,'lightning_indexer_kernel_vec'))
                db.execute('INSERT INTO kernels VALUES (?,?,?,?,?)', (1,pos+2000,pos+12000,10000,'mul_mat_vec_q'))
            db.commit(); db.close()
            report=read_trace(path,'synthetic',993280,None,None)
            self.assertEqual(report['window']['logits_readbacks_in_window'],8)
            self.assertIn('qsa_lightning_indexer', report['families_ms'])
            self.assertEqual(report['gpu_agents'][0]['kernel_dispatches'],16)
            self.assertTrue(report['gpu_agents'][0]['busy_union_window_pct'] > 0)

    def test_missing_views(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'empty.db';sqlite3.connect(p).close()
            with self.assertRaises(ValueError):read_trace(p,'bad',993280,None,None)

if __name__ == '__main__':
    unittest.main()
