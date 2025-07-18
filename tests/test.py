import torch


@torch.no_grad()
def test_dataset(dataset_test, data_path):
    ans = torch.load("./tests/obj/dataset.pt")

    idx_to_test = [0, 35, 70, 71, 100, 134, 135, 136, 4216, 2481]

    for i, el in enumerate(idx_to_test):
        assert torch.all(ans[i] == dataset_test[el][0]), "Test failed.\n"
        break

    print("Test passed")


@torch.no_grad()
def test_action_dec(act_dec):
    act_dec = act_dec.to("cpu")
    x = torch.ones((1, act_dec.latent_act_dim))
    assert act_dec(x).shape == (1, act_dec.true_act_dim), (
        f"Test failed.\nExpected shape: (1, act_dec.true_act_dim), got {act_dec(x).shape}"
    )
    print("Test passed")


@torch.no_grad()
def test_idm(idm):
    idm = idm.to("cpu")
    x = torch.ones((1, 4, 3, 64, 64))
    assert idm(x).shape == (1, 128), (
        f"Test failed.\nExpected shape: (1, 128), got {idm(x).shape}"
    )

    print("Test passed")


@torch.no_grad()
def test_fdm(fdm):
    fdm = fdm.to("cpu")
    x = torch.ones((1, 3, 64, 64))
    lat = torch.ones((1, 128))
    assert fdm(x, lat).shape == (1, 3, 64, 64), "Test failed"
    print("Test passed")


@torch.no_grad()
def test_bc_actor(actor):
    x = torch.zeros((1, 1, 3, 64, 64))
    assert actor(x)[0].shape == (1, actor.num_actions), (
        f"Test failed.\nExpected shape: {(1, actor.num_actions)}, got {actor(x)[0].shape.shape}"
    )
    print("Test passed")


@torch.no_grad()
def test_action_dec(act_dec):
    act_dec = act_dec.to("cpu")
    x = torch.ones((1, act_dec.latent_act_dim))
    assert act_dec(x).shape == (1, act_dec.true_act_dim), "Test failed"
    print("Test passed")
