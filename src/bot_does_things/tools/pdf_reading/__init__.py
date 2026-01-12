try:
    import surya
    import pdfplumber
except:
    raise ModuleNotFoundError(
        "Using the pdf_reading module requires installing bot_does_things[pdf]"
    )
