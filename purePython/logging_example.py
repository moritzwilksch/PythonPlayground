#%%
import logging

logging.basicConfig(
    format="[%(levelname)s] %(asctime)s - %(message)s",
    level=logging.DEBUG,
    handlers=[logging.StreamHandler(), logging.FileHandler("logoutput.log")],
)
logging.debug("Debug")
logging.info("Info")
logging.warning("Warn")
logging.error("Error")

