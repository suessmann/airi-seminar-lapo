import torch

@torch.no_grad()
def test_dataset(dataset_test, data_path):
    ans = torch.load('./tests/obj/dataset.pt')

    idx_to_test = [0, 30, 60, 61, 62, 126, 127, 128]

    for i, el in enumerate(idx_to_test):
        assert torch.all(ans[i] == dataset_test[el][0]), "Test failed"
        break
        
    print('Test passed')

@torch.no_grad()
def test_idm(idm):
    idm = idm.to('cpu')
    x = torch.ones((1, 4, 3, 256, 256))
    assert idm(x).shape == (1, 128), 'Test failed'

    print('Test passed')

@torch.no_grad()
def test_fdm(fdm):
    fdm = fdm.to('cpu')
    x = torch.ones((1, 3, 256, 256))
    lat = torch.ones((1, 128))
    assert fdm(x, lat).shape == (1, 3, 256, 256), 'Test failed'
    print('Test passed')

@torch.no_grad()
def test_bc_actor(actor):
    x = torch.zeros((1, 1, 3, 256, 256))
    assert actor(x)[0].shape == (1, actor.num_actions), "Test failed"
    print("Test passed")

@torch.no_grad()
def test_action_dec(act_dec):
    act_dec = act_dec.to('cpu')
    x = torch.ones((1, act_dec.latent_act_dim))
    assert act_dec(x).shape == (1, act_dec.true_act_dim), 'Test failed'
    print('Test passed')