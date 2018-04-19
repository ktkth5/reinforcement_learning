import retro
import torchvision.transforms as T

from PIL import Image

def get_screen(env):
    """
    :param env:
    :return: Torch.FloatTensor(3,40,40)
    """
    state = env.render(mode="rgb_array").transpose(2,0,1)
    resize = T.Compose([T.ToPILImage(),
                        T.Resize((40,40), interpolation=Image.CUBIC),
                        T.ToTensor()])
    state = resize(state)
    return state

if __name__=="__main__":
    env = retro.make(game='Airstriker-Genesis', state='Level1')
    get_screen(env)