 





<h3 align="center">Emergency Braking Intention Detector</h3>

  <p align="center">
An emergency braking intention detector for motorists. Trained a support vector machine on a dataset which recorded the brain signals of participants in a driving simulation. Participants wore EEG headsets and drove for 20 miutes in a driving simulation. The helmets recorded data streams consisting of 64 electroencephalogram signals. I used supervised learning and a sliding window to segment the streams into windows which were ascribed a label of 1 for 'about to brake' and 0 for 'not about to brake'. I also extracted features for the columns (individual signal) in each sliding glass window. Features included mean, standard deviation, activity, mobility, and complexity. I achieved a 90% true positive rate and a 92% true negative rate after evaluating the model with the test set.
  </p>
</div>

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/dturnover/Emergency-Braking-Intention-Detector
   ```
2. Install Matlab

<!-- USAGE EXAMPLES -->
## Usage

Must be run using Matlab

An EEG dataset consisting of 64 signals must be fed to the program by changing the directory on line 1 to the directory where your data is stored

Link to dataset: https://mega.nz/file/g19BAbiS#8puF48I_66n2o5g1K86evw3QNNjK3SL3EkcoFjSChRs

Link to dataset description: https://lampx.tugraz.at/~bci/database/002-2016/description.txt

Unzip the files into a folder called 'Emergency_Braking_EEG_Data'. Make sure the Matlab file is in the same directory as this folder before running.

<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<!-- CONTACT -->
## Contact

Desmond Turner - desmondegt@gmail.com.com

Project Link: [https://github.com/dturnover/Logistic-Regression-Classifier/](https://github.com/dturnover/Logistic-Regression-Classifier/)

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/dturnover/repo_name.svg?style=for-the-badge
[contributors-url]: https://github.com/dturnover/repo_name/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/dturnover/repo_name.svg?style=for-the-badge
[forks-url]: https://github.com/dturnover/repo_name/network/members
[stars-shield]: https://img.shields.io/github/stars/dturnover/repo_name.svg?style=for-the-badge
[stars-url]: https://github.com/dturnover/repo_name/stargazers
[issues-shield]: https://img.shields.io/github/issues/dturnover/repo_name.svg?style=for-the-badge
[issues-url]: https://github.com/dturnover/repo_name/issues
[license-shield]: https://img.shields.io/github/license/dturnover/repo_name.svg?style=for-the-badge
[license-url]: https://github.com/dturnover/repo_name/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/desmond-turner-006b36191
[product-screenshot]: images/screenshot.png
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com 

