# Notebook

```
def init(filename):
    from bs4 import BeautifulSoup as bf
    import requests, json
    response = requests.get(f"https://gitlab.com/vaisakh678/Notebooks/-/raw/main/{filename}")
    return json.loads(str(bf(response.content, "html.parser")))

txt = init("whole.json"
```

<p>To check notebook is running on google colab or not. </p>

```
def isColab():
    import sys
    if 'google.colab' in sys.modules:
        return True
    return False
```
<p>Basic config..</p>


```
# config..
def isColab():
    import sys
    if 'google.colab' in sys.modules:
        return True
    return False
    
if isColab:
    try:
        from google.colab import drive
        drive.mount("/content/drive")
    except:
        pass
```

<input></input>
