# カルマンフィルターを実装する

## 本リポジトリの趣旨
- カルマンフィルターの実装は世の中に溢れているが、いずれもnumpy形式で一気に処理を行い、全体的に関数に責任を任せているコードが多い。一方で、多くの場合に対して参考書では、各時点ごとの処理を記述されており、初心者にはコードの実装と数式の処理を追うのが大変だと思われる。
- 本リポジトリは、処理の理解のためにコードの速度よりも直感的なわかりやすさを優先するために作成した。たとえば、関連のあるオブジェクトは、1つのデータクラスとしてまとめたり、平滑化を行うための予測やフィルタリング結果の保存を管理するオブジェクトを自作している。
- 作成者本人も勉強のために作成したものであり、間違いを含んでいる可能性があることに注意されたい。

## 内容
- ローカルモデル
- RTS平滑化

## インストール方法
`poetry.lock`と`pyproject.toml`を置いておくので、興味のある方は実行してみてください。
