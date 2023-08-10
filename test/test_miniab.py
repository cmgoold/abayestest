from miniab import MiniAb

def test_miniab_instance():
    ab = MiniAb(force_compile=True)
    assert isinstance(ab, MiniAb)
