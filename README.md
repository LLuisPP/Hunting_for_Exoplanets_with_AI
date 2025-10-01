<div align="center">

  # <a href="#"><img align="center" src="https://github.com/user-attachments/assets/81da15bc-f84e-49f0-9b1c-5ab52ed89a28"></a> <br><br> SWAI - Hunting for Exoplanets with AI
<br>Silent Watcher AI (SWAI) is an interactive project that combines AI with NASA’s open datasets to classify and explore exoplanets. It makes cutting-edge space science accessible through an intuitive web interface, fostering curiosity, learning, and engagement in the search for new worlds beyond our solar system.
</div>

<h2 align="center">Welcome to A World Away: Hunting for Exoplanets with AI. <br> I teamed up with AI to form Team SWAI, from 42 Bcn</h2>


<div align="center">
<table>
  <tr>
    <td align="center" width="300">ChatGPT</td>
    <td align="center" width="300">Gemini</td>
    <td align="center" width="300">Luis Prieto</td>
  </tr>
  <tr>
    <td align="center" width="300"><a href="#"><img width="145" src="https://github.com/user-attachments/assets/82179b5a-efb6-4116-a916-3178f209662d" /></a></td>
    <td align="center" width="300"><a href="#"><img width="155" src="https://github.com/user-attachments/assets/9f1fe198-7ab5-48cd-b4ed-834f05da599a" /></a></td>
    <td align="center" width="300"><a href="#"><img width="153" src="https://github.com/user-attachments/assets/d3ec9f86-88c2-4516-b226-675717a81807" /></a></td>
  </tr>

  <tr>
    <td align="center" width="500">AI model</td>
    <td align="center" width="500">Developer </td>
    <td align="center" width="500">AI model</td>
  </tr>
  <tr>
    <td align="center" width="500">Model: ChatGPT 5.0</td>
    <td align="center" width="500">42 Student: lprieto-</td>
    <td align="center" width="500">Model: NotebookLM</td>
  </tr>
</table>

</div>

# Team logo

<div align="center">

|Logo|Why SWAI?|
  |---|---|
  |<a href="#"><img src="https://github.com/user-attachments/assets/6134b489-1f76-48de-a47b-5deb60949dc7" alt="swai" width="300" height="250"/></a>|The name SWAI stands for Silent Watcher Artificial Intelligence. It suggest the idea of an AI system that quietly and consistently observes stellar data, searching for the faint signals that reveal hidden worlds. By combining space-data observation with machine learning, SWAI identifies subtle patterns in light curves and improves the classification of exoplanets. The concept of the “Silent Watcher” reflects patience, precision, and the pursuit of discovery through science and technology.|


</div>

# Description

<h3 weight="bold">The project focuses on building a machine learning model capable of classifying exoplanets and making the results accessible through an interactive web application.</h3>

The application is designed to run on the web for accessibility.  
It uses real scientific data provided by <b>NASA</b> and <b>Exoplanet Archives</b>.  
The <b>machine learning model</b> performs supervised classification of planetary candidates, confirmed planets, and false positives.  
The <b>web interface</b> enables users to test new data points in real time, visualize outputs, and engage with the classification process. 

<div align="center">
<a href="#"><img width="275" src="https://github.com/user-attachments/assets/ee16c71f-4fe1-4d97-996a-9cec2ba0e44b" /></a>
<a href="#"><img width="300" src="https://github.com/user-attachments/assets/1e41426a-5c35-4ca2-a5b5-513a95c48192" /></a>
</div>

General Specifications:

`````
You may (but are not required to) consider the following:

Target audience: students, researchers, and space enthusiasts interested in exoplanet science.

Your tool could:
• Provide an interactive web interface to classify exoplanets in real time using NASA datasets (Kepler, K2, TESS).
• Show how exoplanets are detected and confirmed, focusing on light curves and transit methods.
• Visualize the difference between candidate planets, confirmed planets, and false positives.
• Explain the role of NASA’s Exoplanet Exploration Program and link to official NASA resources.
• Offer simple educational visualizations to make data science and astronomy accessible.
• Optionally include extra interactive features (e.g., data input by users, exploration of model accuracy, or quizzes to reinforce learning).

For datasets and resources, NASA’s Exoplanet Archive and mission data are integrated directly into the project.
`````

# Tecnologies

IDE's & languages:
<br>
<div align="center">
<table>
  <tr>
    <td align="center" width="300">Technologies</td>
    <td align="center" width="600">Libs & tools<br></td>
  </tr>
  <tr>
    <td align="center">
      <a href="#"><img width="50" src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg" /></a>
      <a href="#"><img width="50" src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/javascript/javascript-original.svg" /></a>
      <a href="#"><img width="50" src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/html5/html5-original.svg" /></a>
      <a href="#"><img width="50" src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/css3/css3-original.svg" /></a>
      <a href="#"><img width="50" src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/git/git-original.svg" /></a>
    </td>
    <td align="center" width="250">
      • TensorFlow / Scikit-learn <br>
      • Pandas / NumPy <br>
      • Matplotlib / Seaborn <br>
      • Flask / Streamlit (for web interface) <br>
      • GitHub Actions (deployment & CI/CD)
    </td>
  </tr>
</table>
</div>

  
</div>

<h3>How to install: <a href="#"><img align="right" src="https://img.shields.io/badge/Instructions-INSTALL-red"></a></h3> 

> [!WARNING]
> You need to install at least blender, and the local server in order to execute the program


`````
1- Donwload repo files.
  A. Install Node.js
    Download and install from nodejs.org.
  B. Install Three.js
    npm install three
  C. Install a local server
    npm install -g http-server

2- Run local https-server in your terminal: if you dont have one on your computer:
    run server certificates :    openssl req -nodes -new -x509 -keyout server.key -out server.cert
    run server:                  http-server -S -C server.cert -K server.key -p 3000 -c-1
3- Load on your browser the net IP shown in the server status in your terminal.
4- If you dont have VR Googles, controlers are shown using VR browser extension.

`````

<h3>Developing tools:</h3>

Python & package manager
https://www.python.org/ | https://pip.pypa.io/<br>
Virtual environments
https://docs.python.org/3/library/venv.html | https://python-poetry.org/<br>
Jupyter Notebook / Lab
https://jupyter.org/<br>
Data analysis
https://pandas.pydata.org/ | https://numpy.org/<br>
Machine Learning
https://scikit-learn.org/ | https://www.tensorflow.org/ (opcional)<br>
Visualization
https://matplotlib.org/<br>
Web app (frontend ligero + backend)
https://streamlit.io/ | https://flask.palletsprojects.com/ (alternativa)<br>
Model persistence (opcional)
https://joblib.readthedocs.io/en/latest/ | https://docs.python.org/3/library/pickle.html<br>
Linting & formatting
https://docs.astral.sh/ruff/ | https://black.readthedocs.io/<br>
Gimp for image edit https://www.gimp.org/



https://transfer.zip/#NPz9iWVB7ixCui4joykbDvnNYF9nmhztx3DvR7KgKO0,aaebfd37-0d00-4cec-81fa-0ff16557916d,R

<h2>Webgraphy</h2>

<b>Documentation & Open-Source Principles</b><br>
- https://k12cs.org/navigating-the-practices/ <br>
- Principle of Citation/Code Use: Incorporate existing code, media, and libraries into original programs while citing their source. <br>
- Use of Digital Tools: Employ digital tools (e.g., computers) to analyze very large datasets for patterns and trends. <br>

<b>Scientific Data & Exoplanet Context</b><br>
- https://exoplanetarchive.ipac.caltech.edu/ <br>
- https://es.wikipedia.org/wiki/Planeta_extrasolar <br>
- https://es.wikipedia.org/wiki/M%C3%A9todos_de_detecci%C3%B3n_de_planetas_extrasolares <br>
- https://www.esa.int/Science_Exploration/Space_Science/Cheops/How_to_find_an_exoplanet <br>

<b>Hackathon & NASA Resources</b><br>
- https://www.spaceappschallenge.org/ <br>
- https://exoplanets.nasa.gov/ <br>

---

<h2>Document-PDF</h2>
https://docs.google.com/document/d/1YJhD04ney8mo_zYtPxs_bnenb_9mb492uUlvESLh5YI/edit?usp=sharing  

<h3>Vídeo</h3>
https://drive.google.com/file/d/16cv5urkByHFgwvqxFyNR9PsIOqHtBf5H/view?usp=sharing  

<br>
<h2>Participation Certificate</h2>
<div align="center">
<a href="#"><img width="700" src="https://github.com/user-attachments/assets/aa764ce1-18b4-4631-b12b-8b637774817f" /></a>
</div>
