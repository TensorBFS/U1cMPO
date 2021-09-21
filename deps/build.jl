# see https://gist.github.com/Luthaf/368a23981c8ec095c3eb
import PyCall: pyimport

# See https://stackoverflow.com/questions/12332975/installing-python-module-within-code.
const PIP_PACKAGES = ["py3nj"]

sys = pyimport("sys")
subprocess = pyimport("subprocess")
subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", "--upgrade", "--force-reinstall", PIP_PACKAGES...])